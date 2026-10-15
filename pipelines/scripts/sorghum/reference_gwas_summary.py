"""Compare the GWAS loci of the sorghum H2-opt traits with those of the reference traits.

The loci are the Bonferroni hits of the H2-opt traits (gwas.py, all --h2opt files and traits)
grouped by LD clumping (gwas.clump: r^2 above --r2 with the lead SNP within --kb kilobases, LD
from the genotypes --genotypes). For each locus and each reference trait (reference_gwas.py), the
largest -log10 p within --window bp of the locus. Also the genome-wide Bonferroni hits, largest
-log10 p and lambda GC of each reference trait. Writes a .tsv (one row per reference trait) and
prints it.

    python pipelines/scripts/sorghum/reference_gwas_summary.py --h2opt h2opt_fold*.npz \\
        --genotypes snps --r2 0.5 --kb 250 \\
        --reference reference/ndvi.npz reference/height_joint_model.npz \\
        --window 250000 --out reference/summary.tsv
"""

import argparse
from pathlib import Path

import numpy as np

from h2opt_analysis import gwas


def loci(h2opt_files, genotypes, r2, kb):
    """(chr, first, last, lead, -log10 p of the lead, the files and traits with a hit) of each
    locus of the Bonferroni hits of the H2-opt files, by position."""
    scores, labels = [], []
    for path in h2opt_files:
        with np.load(path) as f:
            chrom, ps = f['chr'], f['ps']
            scores.append(f['neg_log10_p'])
            labels += [f'{Path(path).stem} trait {t + 1}' for t in range(len(f['neg_log10_p']))]
    score = np.concatenate(scores)
    groups = []
    for locus in gwas.clump(genotypes, chrom, ps, score, r2=r2, kb=kb):
        lead, members = locus['lead'], locus['snps']
        groups.append((int(chrom[lead]), int(ps[members].min()), int(ps[members].max()),
                       int(ps[lead]), float(score[:, lead].max()),
                       [labels[r] for r in locus['rows']]))
    return sorted(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h2opt', nargs='+', required=True, help='gwas.py outputs (.npz)')
    parser.add_argument('--reference', nargs='+', required=True,
                        help='reference_gwas.py outputs (.npz), named <trait>.npz')
    parser.add_argument('--genotypes', required=True, help='PLINK set (prefix) for the LD')
    parser.add_argument('--r2', type=float, required=True, help='LD clumping r^2 threshold')
    parser.add_argument('--kb', type=int, required=True, help='LD clumping window (kb)')
    parser.add_argument('--window', type=int, required=True, help='bp on each side of a locus')
    parser.add_argument('--out', required=True, help='.tsv')
    args = parser.parse_args()

    groups = loci(args.h2opt, args.genotypes, args.r2, args.kb)
    print(f'H2-opt loci ({len(args.h2opt)} GWAS files; LD clumping, r^2 > {args.r2} within '
          f'{args.kb} kb):')
    for c, first, last, lead, score, hits in groups:
        print(f'  chr{c}:{first}-{last}, lead {lead} (-log10 p {score:.2f}): {", ".join(hits)}')
    columns = [f'chr{c}:{first}-{last}' for c, first, last, *_ in groups]
    header = ['trait', 'bonferroni_hits', 'max_neg_log10_p', 'lambda_gc'] + columns
    rows = []
    for path in args.reference:
        with np.load(path) as f:
            score, chrom, ps, lam = f['neg_log10_p'][0], f['chr'], f['ps'], f['lambda_gc'][0]
        near = [score[(chrom == c) & (ps >= first - args.window) & (ps <= last + args.window)].max()
                for c, first, last, *_ in groups]
        hits = gwas.bonferroni_hits(score[None])[0]
        rows.append([Path(path).stem, str(hits), f'{score.max():.2f}', f'{lam:.3f}']
                    + [f'{s:.2f}' for s in near])
    threshold = -np.log10(0.05 / len(score))
    print(f'\nlargest -log10 p within {args.window} bp of each locus (Bonferroni {threshold:.2f}):')
    lines = ['\t'.join(header)] + ['\t'.join(r) for r in rows]
    print('\n'.join(lines))
    Path(args.out).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
