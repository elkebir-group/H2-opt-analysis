"""GWAS with GEMMA's univariate linear mixed model (-lmm 1, Wald test) on a PLINK genotype set.

Phenotypes are given per individual in .fam order; NaN marks a missing phenotype. Individuals with
missing phenotypes are removed from a copy of the .bed/.fam in a working directory rather than
marked NA in the .fam: GEMMA 0.97 reports wrong per-SNP effects and p-values (with the right null
model) when the .fam has NA phenotypes. The original files are never changed. GEMMA adds no
intercept when covariates are given (-c), so run_gemma adds one unless told otherwise.

The GEMMA binary is $GEMMA, or bin/gemma-0.97/gemma in this repository (GEMMA 0.97, the version used
for the paper). GEMMA 0.97 fails when the number of SNPs analyzed is a multiple of 10,000 (its batch
size).
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .paths import REPO

ASSOC_FIELDS = ['beta', 'se', 'l_remle', 'p_wald']


def gemma_binary():
    """Path of the GEMMA binary."""
    return os.environ.get('GEMMA', str(REPO / 'bin' / 'gemma-0.97' / 'gemma'))


def read_fam(prefix):
    """Family and individual IDs of the .fam file, as an (n, 2) string array."""
    return np.loadtxt(f'{prefix}.fam', dtype=str, usecols=(0, 1), ndmin=2)


def read_bed(prefix, snps=None):
    """Genotypes as counts of allele 1 (individuals x SNPs), NaN where missing.

    snps selects SNP rows of the .bim.
    """
    n = len(read_fam(prefix))
    with open(f'{prefix}.bed', 'rb') as f:
        assert f.read(3) == b'\x6c\x1b\x01', 'expected a SNP-major PLINK .bed file'
        raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, (n + 3) // 4)
    if snps is not None:
        raw = raw[snps]
    codes = (raw[:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    codes = codes.reshape(len(raw), -1)[:, :n]
    # PLINK codes: 0 = homozygous allele 1, 1 = missing, 2 = heterozygous, 3 = homozygous allele 2
    return np.array([2, np.nan, 1, 0], dtype=np.float32)[codes].T


def genotype_pcs(prefix, n_pcs=5, individuals=None, chunk=20000):
    """Top principal components of the SNP-centered genotypes, each scaled to unit root mean square.

    Missing genotypes are set to the SNP mean. individuals (boolean, .fam order) selects the
    individuals; the others get NaN.
    """
    n = len(read_fam(prefix))
    keep = np.ones(n, bool) if individuals is None else np.asarray(individuals, bool)
    m = sum(1 for _ in open(f'{prefix}.bim'))
    gram = np.zeros((keep.sum(), keep.sum()))
    for start in range(0, m, chunk):
        G = read_bed(prefix, slice(start, start + chunk))[keep].astype(np.float64)
        G -= np.nanmean(G, axis=0)
        G[np.isnan(G)] = 0
        gram += G @ G.T
    _, vectors = np.linalg.eigh(gram)
    pcs = vectors[:, ::-1][:, :n_pcs]
    pcs /= np.sqrt(np.mean(pcs ** 2, axis=0))
    out = np.full((n, n_pcs), np.nan)
    out[keep] = pcs
    return out


def write_bed_subset(prefix, keep, out_prefix, chunk=20000):
    """Copy of the PLINK set with only the individuals where keep (boolean, .fam order) is true.

    The .bim is symlinked.
    """
    keep = np.asarray(keep, bool)
    n, out_prefix = len(keep), Path(out_prefix)
    in_bytes, pad = (n + 3) // 4, (-keep.sum()) % 4
    with open(f'{prefix}.bed', 'rb') as src, open(f'{out_prefix}.bed', 'wb') as dst:
        assert src.read(3) == b'\x6c\x1b\x01', 'expected a SNP-major PLINK .bed file'
        dst.write(b'\x6c\x1b\x01')
        while block := src.read(in_bytes * chunk):
            raw = np.frombuffer(block, dtype=np.uint8).reshape(-1, in_bytes)
            codes = (raw[:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
            codes = codes.reshape(len(raw), -1)
            codes = np.pad(codes[:, :n][:, keep], ((0, 0), (0, pad))).reshape(len(raw), -1, 4)
            packed = codes[..., 0] | codes[..., 1] << 2 | codes[..., 2] << 4 | codes[..., 3] << 6
            dst.write(packed.tobytes())
    link = Path(f'{out_prefix}.bim')
    link.unlink(missing_ok=True)
    link.symlink_to(Path(f'{prefix}.bim').resolve())


def write_snp_subset(prefix, snps, out_prefix, chunk=20000):
    """Complete PLINK set (.bed, .bim, .fam) with only the SNPs where snps (boolean, .bim order) is
    true; all individuals are kept."""
    snps = np.asarray(snps, bool)
    n = len(read_fam(prefix))
    row_bytes = (n + 3) // 4
    with open(f'{prefix}.bed', 'rb') as src, open(f'{out_prefix}.bed', 'wb') as dst:
        assert src.read(3) == b'\x6c\x1b\x01', 'expected a SNP-major PLINK .bed file'
        dst.write(b'\x6c\x1b\x01')
        start = 0
        while block := src.read(row_bytes * chunk):
            raw = np.frombuffer(block, dtype=np.uint8).reshape(-1, row_bytes)
            dst.write(raw[snps[start:start + len(raw)]].tobytes())
            start += len(raw)
    if start != len(snps):
        raise ValueError(f'{len(snps)} SNP flags for {start} SNPs in {prefix}.bed')
    with open(f'{prefix}.bim') as src, open(f'{out_prefix}.bim', 'w') as dst:
        dst.writelines(line for line, kept in zip(src, snps, strict=True) if kept)
    shutil.copyfile(f'{prefix}.fam', f'{out_prefix}.fam')


def run_gemma(prefix, phenotypes, workdir, covariates=None, intercept=True, jobs=1, threads=1,
              binary=None):
    """GEMMA LMM association of each phenotype column with every SNP.

    prefix: PLINK genotype set (path without .bed). phenotypes: (n,) or (n, traits) in .fam order,
    NaN = missing; all columns must have the same missing individuals, which are left out (one
    relatedness matrix, on the individuals with phenotypes, serves all traits). covariates: (n, c)
    in .fam order or None; an intercept column is put first unless intercept=False. Traits run jobs
    at a time, each GEMMA process with threads OpenBLAS threads. Returns a dict: chr, rs, ps,
    allele1, allele0, af (per SNP) and beta, se, l_remle, p_wald (traits x SNPs).
    """
    prefix, workdir = Path(prefix).resolve(), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    binary = binary or gemma_binary()
    Y = np.asarray(phenotypes, dtype=np.float64).reshape(len(phenotypes), -1)
    missing = np.isnan(Y)
    if not (missing == missing[:, :1]).all():
        raise ValueError('all phenotypes must have the same missing individuals')
    fam = np.loadtxt(f'{prefix}.fam', dtype=str, usecols=range(5), ndmin=2)
    if len(fam) != len(Y):
        raise ValueError(f'{len(Y)} phenotype rows for {len(fam)} individuals in {prefix}.fam')

    keep = ~missing[:, 0]
    if keep.all():
        for ext in ['bed', 'bim']:
            link = workdir / f'geno.{ext}'
            link.unlink(missing_ok=True)
            link.symlink_to(f'{prefix}.{ext}')
    else:
        write_bed_subset(prefix, keep, workdir / 'geno')
    fam_rows = np.hstack([fam[keep], np.char.mod('%.10g', Y[keep])])
    np.savetxt(workdir / 'geno.fam', fam_rows, fmt='%s', delimiter=' ')
    args = []
    if covariates is not None or intercept:
        if covariates is None:
            W = np.ones((len(Y), 0))
        else:
            W = np.asarray(covariates, dtype=np.float64).reshape(len(Y), -1)
        if intercept:
            W = np.hstack([np.ones((len(Y), 1)), W])
        if np.isnan(W[keep]).any():
            raise ValueError('missing covariates for individuals with phenotypes')
        np.savetxt(workdir / 'covariates.txt', W[keep], fmt='%.17g', delimiter='\t')
        args = ['-c', 'covariates.txt']

    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    def gemma(*options, log):
        with open(workdir / log, 'w') as out:
            return subprocess.Popen([binary, '-bfile', 'geno', '-outdir', 'output', *options],
                                    cwd=workdir, stdout=out, stderr=subprocess.STDOUT, env=env)

    def check(process, log):
        if process.wait() != 0:
            raise RuntimeError(f'GEMMA failed (exit {process.returncode}); see {workdir / log}')

    check(gemma('-gk', '1', '-o', 'relatedness', log='relatedness.log'), 'relatedness.log')
    pending, running = list(range(Y.shape[1])), []
    while pending or running:
        while pending and len(running) < jobs:
            t = pending.pop(0)
            log = f'trait{t + 1}.log'
            process = gemma('-k', 'output/relatedness.cXX.txt', *args, '-lmm', '1',
                            '-n', str(t + 1), '-o', f'trait{t + 1}', log=log)
            running.append((process, log))
        check(*running.pop(0))
    results = [read_assoc(workdir / 'output' / f'trait{t + 1}.assoc.txt')
               for t in range(Y.shape[1])]
    for r in results[1:]:
        assert (r['rs'] == results[0]['rs']).all()
    out = {k: v for k, v in results[0].items() if k not in ASSOC_FIELDS}
    out.update({k: np.stack([r[k] for r in results]) for k in ASSOC_FIELDS})
    return out


def read_assoc(path):
    """A GEMMA .assoc.txt file as a dict of arrays (strings for chr, rs, alleles; numbers else)."""
    table = np.loadtxt(path, dtype=str, delimiter='\t', ndmin=2)
    header, rows = table[0], table[1:]
    text = {'chr', 'rs', 'allele1', 'allele0'}
    integer = {'ps', 'n_miss'}

    def column(i, name):
        if name in text:
            return rows[:, i]
        return rows[:, i].astype(np.int64 if name in integer else float)

    return {name: column(i, name) for i, name in enumerate(header)}
