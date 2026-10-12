"""Compare the GWAS loci of the sorghum H2-opt traits with those of the reference traits.

A locus is a group of the Bonferroni hits of the H2-opt traits (gwas.py, all --h2opt files and
traits) on one chromosome, each hit within --merge bp of the next. For each locus and each
reference trait (reference_gwas.py), the largest -log10 p within --window bp of the locus. Also
the genome-wide Bonferroni hits, largest -log10 p and lambda GC of each reference trait. Writes a
.tsv (one row per reference trait) and prints it.

    python pipelines/scripts/sorghum/reference_gwas_summary.py --h2opt h2opt_fold*.npz \\
        --reference reference/ndvi.npz reference/height_joint_model.npz \\
        --merge 1000000 --window 250000 --out reference/summary.tsv
"""

import argparse
from pathlib import Path

import numpy as np

from h2opt_analysis import gwas


def loci(h2opt_files, merge):
    """(chr, first, last, hits, files) of each locus of the Bonferroni hits of the H2-opt files."""
    hits = []
    for i, path in enumerate(h2opt_files):
        with np.load(path) as f:
            score, chrom, ps = f['neg_log10_p'], f['chr'], f['ps']
        threshold = -np.log10(0.05 / score.shape[1])
        for _, j in zip(*np.nonzero(score > threshold), strict=True):
            hits.append((int(chrom[j]), int(ps[j]), i))
    groups = []
    for c, p, i in sorted(hits):
        if groups and groups[-1][0] == c and p - groups[-1][2] <= merge:
            groups[-1][2] = p
            groups[-1][3] += 1
            groups[-1][4].add(i)
        else:
            groups.append([c, p, p, 1, {i}])
    return [(c, first, last, n, len(files)) for c, first, last, n, files in groups]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h2opt', nargs='+', required=True, help='gwas.py outputs (.npz)')
    parser.add_argument('--reference', nargs='+', required=True,
                        help='reference_gwas.py outputs (.npz), named <trait>.npz')
    parser.add_argument('--merge', type=int, required=True,
                        help='largest gap in bp between two hits of one locus')
    parser.add_argument('--window', type=int, required=True, help='bp on each side of a locus')
    parser.add_argument('--out', required=True, help='.tsv')
    args = parser.parse_args()

    groups = loci(args.h2opt, args.merge)
    print(f'H2-opt loci ({len(args.h2opt)} GWAS files; hits within {args.merge} bp merged):')
    for c, first, last, n, files in groups:
        print(f'  chr{c}:{first}-{last}: {n} hits in {files} files')
    columns = [f'chr{c}:{first}-{last}' for c, first, last, _, _ in groups]
    header = ['trait', 'bonferroni_hits', 'max_neg_log10_p', 'lambda_gc'] + columns
    rows = []
    for path in args.reference:
        with np.load(path) as f:
            score, chrom, ps, lam = f['neg_log10_p'][0], f['chr'], f['ps'], f['lambda_gc'][0]
        near = [score[(chrom == c) & (ps >= first - args.window) & (ps <= last + args.window)].max()
                for c, first, last, _, _ in groups]
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
