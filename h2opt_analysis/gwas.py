"""GWAS with GEMMA's univariate linear mixed model (-lmm 1, Wald test) on a PLINK genotype set.

Phenotypes are given per individual in .fam order; NaN marks a missing phenotype. Individuals with
missing phenotypes are removed from a copy of the .bed/.fam in a working directory rather than
marked NA in the .fam: GEMMA 0.97 reports wrong per-SNP effects and p-values (with the right null
model) when the .fam has NA phenotypes. The original files are never changed. GEMMA adds no
intercept when covariates are given (-c), so run_gemma adds one unless told otherwise.

The GEMMA binary is $GEMMA, or bin/gemma-0.97/gemma in this repository (GEMMA 0.97).
GEMMA 0.97 fails when the number of SNPs analyzed is a multiple of 10,000 (its batch
size).

Every file written here first replaces whatever is at its path, so a write never goes through a
symlink that an earlier call left there into the source genotype files (new_file).
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats

from .paths import REPO

ASSOC_FIELDS = ['beta', 'se', 'l_remle', 'p_wald']
BED_MAGIC = b'\x6c\x1b\x01'  # SNP-major PLINK .bed
# Genotype of each 2-bit PLINK code, as the count of allele 1: 0 = homozygous allele 1,
# 1 = missing, 2 = heterozygous, 3 = homozygous allele 2.
DOSAGE = np.array([2, np.nan, 1, 0], dtype=np.float32)


def gemma_binary():
    """Path of the GEMMA binary."""
    return os.environ.get('GEMMA', str(REPO / 'bin' / 'gemma-0.97' / 'gemma'))


def read_fam(prefix):
    """Family and individual IDs of the .fam file, as an (n, 2) string array."""
    return np.loadtxt(f'{prefix}.fam', dtype=str, usecols=(0, 1), ndmin=2)


def read_bim(prefix):
    """The .bim file as an (m, 6) string array: chromosome, ID, cM, position, allele 1, allele 2."""
    return np.loadtxt(f'{prefix}.bim', dtype=str, ndmin=2)


def check_plink_set(prefix):
    """Raise unless the .bed has the size that the .fam and .bim give: 3 + ceil(n / 4) bytes per
    SNP. GEMMA does not check this; with a .fam of the wrong individuals it reads shifted rows."""
    n, m = len(read_fam(prefix)), len(read_bim(prefix))
    size = Path(f'{prefix}.bed').stat().st_size
    if size != 3 + (n + 3) // 4 * m:
        raise ValueError(f'{prefix}.bed has {size} bytes, not 3 + {(n + 3) // 4} x {m} for the '
                         f'{n} individuals of the .fam and {m} SNPs of the .bim')


def new_file(path):
    """path, with any file or symlink at it removed. Write output files only through this: opening
    a leftover symlink for writing would overwrite the file it points to (such as a source .bed)."""
    path = Path(path)
    path.unlink(missing_ok=True)
    return path


def write_plink_set(prefix, fam, out_prefix):
    """PLINK set out_prefix with the .bed of prefix (symlinked), its .bim with every chromosome
    that is not an integer (e.g. unplaced scaffolds) set to 0, and the .fam file fam."""
    out_prefix = Path(out_prefix)
    new_file(f'{out_prefix}.bed').symlink_to(Path(f'{prefix}.bed').resolve())
    bim = read_bim(prefix)
    bim[~np.char.isdigit(bim[:, 0]), 0] = '0'
    np.savetxt(new_file(f'{out_prefix}.bim'), bim, fmt='%s', delimiter='\t')
    shutil.copyfile(fam, new_file(f'{out_prefix}.fam'))
    check_plink_set(out_prefix)


def _bed_blocks(prefix, n, chunk=20000):
    """Raw SNP rows of the .bed file of n individuals, up to chunk SNPs at a time, each an
    (SNPs, bytes per SNP) uint8 array."""
    row_bytes = (n + 3) // 4
    with open(f'{prefix}.bed', 'rb') as f:
        if f.read(3) != BED_MAGIC:
            raise ValueError(f'{prefix}.bed is not a SNP-major PLINK .bed file')
        while block := f.read(row_bytes * chunk):
            yield np.frombuffer(block, dtype=np.uint8).reshape(-1, row_bytes)


def _unpack(raw, n):
    """2-bit codes (SNPs x n) of raw .bed rows."""
    codes = (raw[:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    return codes.reshape(len(raw), -1)[:, :n]


def _pack(codes):
    """Raw .bed rows of 2-bit codes (SNPs x individuals)."""
    codes = np.pad(codes, ((0, 0), (0, (-codes.shape[1]) % 4))).reshape(len(codes), -1, 4)
    return (codes[..., 0] | codes[..., 1] << 2 | codes[..., 2] << 4 | codes[..., 3] << 6)


def read_bed(prefix, snps=None):
    """Genotypes as counts of allele 1 (individuals x SNPs), NaN where missing.

    snps selects SNP rows of the .bim.
    """
    n = len(read_fam(prefix))
    raw = np.concatenate(list(_bed_blocks(prefix, n)))
    if snps is not None:
        raw = raw[snps]
    return DOSAGE[_unpack(raw, n)].T


def genotype_pcs(prefix, n_pcs=5, individuals=None, chunk=20000):
    """Top principal components of the SNP-centered genotypes, each scaled to unit root mean square.

    Missing genotypes are set to the SNP mean. individuals (boolean, .fam order) selects the
    individuals; the others get NaN.
    """
    check_plink_set(prefix)
    n = len(read_fam(prefix))
    keep = np.ones(n, bool) if individuals is None else np.asarray(individuals, bool)
    gram = np.zeros((keep.sum(), keep.sum()))
    for raw in _bed_blocks(prefix, n, chunk):
        G = DOSAGE[_unpack(raw, n)].T[keep].astype(np.float64)
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
    with open(new_file(f'{out_prefix}.bed'), 'wb') as dst:
        dst.write(BED_MAGIC)
        for raw in _bed_blocks(prefix, len(keep), chunk):
            dst.write(_pack(_unpack(raw, len(keep))[:, keep]).tobytes())
    new_file(f'{out_prefix}.bim').symlink_to(Path(f'{prefix}.bim').resolve())


def write_snp_subset(prefix, snps, out_prefix, chunk=20000):
    """Complete PLINK set (.bed, .bim, .fam) with only the SNPs where snps (boolean, .bim order) is
    true; all individuals are kept."""
    snps = np.asarray(snps, bool)
    with open(f'{prefix}.bim') as src:
        n_snps = sum(1 for _ in src)
    if n_snps != len(snps):
        raise ValueError(f'{len(snps)} SNP flags for {n_snps} SNPs in {prefix}.bim')
    start = 0
    with open(new_file(f'{out_prefix}.bed'), 'wb') as dst:
        dst.write(BED_MAGIC)
        for raw in _bed_blocks(prefix, len(read_fam(prefix)), chunk):
            dst.write(raw[snps[start:start + len(raw)]].tobytes())
            start += len(raw)
    with open(f'{prefix}.bim') as src, open(new_file(f'{out_prefix}.bim'), 'w') as dst:
        dst.writelines(line for line, kept in zip(src, snps, strict=True) if kept)
    shutil.copyfile(f'{prefix}.fam', new_file(f'{out_prefix}.fam'))


def line_means(Y, line_of_row, n_lines):
    """Phenotypes for GWAS: the mean of each standardized column of Y over the rows of each line,
    rank-based inverse normal transformed over the lines (rank_normal).

    Y is standardized over all its rows. line_of_row is the .fam row of the line of each row of Y,
    -1 for a row of no line. Returns (n_lines, k) in .fam order; lines without rows get NaN.
    """
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    line_of_row = np.asarray(line_of_row)
    on_line = line_of_row >= 0
    sums = np.zeros((n_lines, Y.shape[1]))
    np.add.at(sums, line_of_row[on_line], Y[on_line])
    counts = np.bincount(line_of_row[on_line], minlength=n_lines)
    with np.errstate(invalid='ignore'):
        return rank_normal(sums / counts[:, None])


def rank_normal(P):
    """Rank-based inverse normal transform of each column of P over its non-NaN rows (Blom offset:
    the normal quantile of (rank - 3/8) / (n + 1/4)). H2-opt traits can have a few extreme lines
    (overfitting on training lines); their rare alleles can then give false GWAS hits. NaN
    stays NaN."""
    P = np.array(P, dtype=np.float64)
    for t in range(P.shape[1]):
        ok = ~np.isnan(P[:, t])
        P[ok, t] = stats.norm.ppf((stats.rankdata(P[ok, t]) - 0.375) / (ok.sum() + 0.25))
    return P


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
    check_plink_set(prefix)
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
            new_file(workdir / f'geno.{ext}').symlink_to(f'{prefix}.{ext}')
    else:
        write_bed_subset(prefix, keep, workdir / 'geno')
    fam_rows = np.hstack([fam[keep], np.char.mod('%.10g', Y[keep])])
    np.savetxt(new_file(workdir / 'geno.fam'), fam_rows, fmt='%s', delimiter=' ')
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
        np.savetxt(new_file(workdir / 'covariates.txt'), W[keep], fmt='%.17g', delimiter='\t')
        args = ['-c', 'covariates.txt']

    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    def gemma(*options, log):
        with open(new_file(workdir / log), 'w') as out:
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


def summarize(result):
    """The saved form of a run_gemma result: chr and ps (per SNP), neg_log10_p (traits x SNPs,
    float32) and lambda_gc, the genomic inflation factor of each trait."""
    p = result['p_wald']
    return {'chr': result['chr'].astype(np.int16), 'ps': result['ps'].astype(np.int64),
            'neg_log10_p': (-np.log10(p)).astype(np.float32),
            'lambda_gc': np.median(stats.chi2.isf(p, 1), axis=1) / stats.chi2.ppf(0.5, 1)}


def bonferroni_hits(neg_log10_p):
    """The number of SNPs of each trait (row of neg_log10_p) with p below 0.05 / the number of
    SNPs."""
    return np.sum(neg_log10_p > -np.log10(0.05 / neg_log10_p.shape[1]), axis=1)


def bim_rows(prefix, chrom, ps):
    """The .bim row of each SNP given by chromosome and position; an error if a SNP is missing
    or ambiguous (two .bim rows with its chromosome and position)."""
    bim = read_bim(prefix)
    rows = {}
    for r, key in enumerate(zip(bim[:, 0].astype(int), bim[:, 3].astype(np.int64), strict=True)):
        rows.setdefault(key, []).append(r)
    found = [rows.get((int(c), int(p)), []) for c, p in zip(chrom, ps, strict=True)]
    bad = [(int(c), int(p)) for c, p, f in zip(chrom, ps, found, strict=True) if len(f) != 1]
    if bad:
        raise ValueError(f'SNPs missing or ambiguous in {prefix}.bim: {bad[:5]}')
    return np.array([f[0] for f in found], dtype=np.int64)


def clump(prefix, chrom, ps, neg_log10_p, r2=0.5, kb=250):
    """Loci of the Bonferroni hits by LD clumping, as PLINK --clump with p1 = p2 = 0.05 / SNPs.

    neg_log10_p: (rows, SNPs), e.g. the traits of several folds stacked, all of the same SNPs
    (chrom, ps); each SNP takes its largest value over the rows. The strongest remaining hit is
    the lead SNP of a new locus, and every remaining hit on its chromosome within kb kilobases and
    with r^2 above r2 (genotypes of the PLINK set prefix, pairwise complete) joins it. Chromosome
    0 holds the SNPs of unknown or unplaced position (e.g. the scaffolds of write_plink_set), so a
    hit on it is a locus of its own. The SNPs are found in the .bim by chromosome and position:
    GEMMA leaves out some SNPs of the set, so an index into its results is not a .bim row. Returns
    the loci, strongest first, as dicts: lead (SNP index), snps (SNP indices) and rows (the rows
    with a hit at a SNP of the locus); no loci without hits."""
    neg_log10_p = np.atleast_2d(neg_log10_p)
    hit = neg_log10_p > -np.log10(0.05 / neg_log10_p.shape[1])
    snps = np.flatnonzero(hit.any(axis=0))
    if not len(snps):
        return []
    G = read_bed(prefix, bim_rows(prefix, chrom[snps], ps[snps]))
    hit_chrom = np.asarray(chrom)[snps].astype(int)
    hit_ps = np.asarray(ps)[snps].astype(np.int64)
    best = neg_log10_p[:, snps].max(axis=0)
    remaining = np.argsort(-best)
    loci = []
    while len(remaining):
        lead = remaining[0]
        near = remaining[(hit_chrom[remaining] == hit_chrom[lead]) & (hit_chrom[lead] != 0)
                         & (np.abs(hit_ps[remaining] - hit_ps[lead]) <= kb * 1000)]
        x = G[:, lead]
        members = [lead]
        for k in near[near != lead]:
            y = G[:, k]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if np.corrcoef(x[ok], y[ok])[0, 1] ** 2 > r2:
                members.append(k)
        loci.append({'lead': snps[lead], 'snps': snps[members],
                     'rows': np.flatnonzero(hit[:, snps[members]].any(axis=1))})
        remaining = remaining[~np.isin(remaining, members)]
    return loci


def report(summary, names=None):
    """Print a summarize result, one line per trait: the number of SNPs, the largest -log10 p, the
    Bonferroni hits and lambda GC. names: the label of each trait (default 'trait 1', ...)."""
    score = summary['neg_log10_p']
    names = names or [f'trait {t + 1}' for t in range(len(score))]
    for name, s, hits, lam in zip(names, score, bonferroni_hits(score), summary['lambda_gc'],
                                  strict=True):
        print(f'{name}: {len(s)} SNPs, max -log10 p {s.max():.2f}, {hits} Bonferroni hits, '
              f'lambda GC {lam:.3f}', flush=True)


def cross_fit(traits, fold_of_row):
    """Cross-fitted trait: the trait of each row from the fold model that held that row out.

    traits: (folds, n), one trait of all rows from each fold's model (e.g. trait 1). Each fold's
    trait is standardized over all rows, and its sign set so that it correlates positively with
    fold 0's (a trait's sign is arbitrary). fold_of_row: (n,) the held-out fold of each row.
    Returns the (n,) cross-fitted trait and the correlation of each fold's trait with fold 0's
    over all rows, after the sign change: the traits must agree for the stitching to make sense."""
    z = np.asarray(traits, dtype=np.float64)
    z = (z - z.mean(axis=1, keepdims=True)) / z.std(axis=1, keepdims=True)
    r = np.array([np.corrcoef(z[0], t)[0, 1] for t in z])
    z = z * np.where(r < 0, -1.0, 1.0)[:, None]
    fold_of_row = np.asarray(fold_of_row)
    return z[fold_of_row, np.arange(z.shape[1])], np.abs(r)
