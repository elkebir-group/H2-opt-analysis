"""GWAS figures of the paper: agreement between the folds (crossGWAS) and Manhattan and QQ plots
of one fold (sorGWAS).

Reads the GWAS .npz of each fold (gwas.py, gwas.summarize format). crossGWAS: the Pearson
correlation of -log10 p of --trait over all SNPs between each pair of folds (a heatmap); the
p-value does not depend on the arbitrary sign of a trait. sorGWAS: Manhattan and QQ plots
(figures.plot_gwas) of the traits of --example-fold with Bonferroni-significant SNPs
(0.05 / SNPs); if no trait has one, of all traits. Writes each figure as .pdf and, beside it,
.png, and prints the numbers.

    python pipelines/scripts/sorghum/plot_paper_gwas.py --folds 0 ... 9 \\
        --gwas gwas/h2opt_fold0.npz ... gwas/h2opt_fold9.npz --trait 1 --example-fold 0 \\
        --out-folds figures/paper/crossGWAS.pdf --out-manhattan figures/paper/sorGWAS.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis.figures import plot_gwas  # noqa: E402


def plot_fold_agreement(scores, folds, trait, path):
    """Heatmap of the Pearson correlation of -log10 p between the folds; returns the matrix."""
    correlation = np.corrcoef(scores)
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    image = ax.imshow(correlation, cmap='magma')
    middle = (correlation.min() + correlation.max()) / 2
    for (i, j), value in np.ndenumerate(correlation):
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=7,
                color='black' if value > middle else 'white')
    labels = [str(f + 1) for f in folds]  # folds as numbered in the paper (1-10)
    ax.set_xticks(range(len(folds)), labels)
    ax.set_yticks(range(len(folds)), labels)
    ax.set_xlabel('cross-validation split')
    ax.set_ylabel('cross-validation split')
    off = correlation[~np.eye(len(folds), dtype=bool)]
    ax.set_title(f'-log10 p of trait {trait}: Pearson r between splits '
                 f'(median {np.median(off):.3f})', fontsize=8)
    fig.colorbar(image, ax=ax, label='Pearson r')
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix('.png'), dpi=150)
    plt.close(fig)
    return correlation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', required=True)
    parser.add_argument('--gwas', nargs='+', required=True, help='one .npz per fold')
    parser.add_argument('--trait', type=int, required=True, help='trait compared across folds')
    parser.add_argument('--example-fold', type=int, required=True)
    parser.add_argument('--out-folds', required=True, help='.pdf; a .png is written beside it')
    parser.add_argument('--out-manhattan', required=True, help='.pdf; a .png beside it')
    args = parser.parse_args()
    if len(args.folds) != len(args.gwas):
        parser.error('give one GWAS file per fold')

    summaries = [np.load(path) for path in args.gwas]
    for s in summaries[1:]:
        assert np.array_equal(s['ps'], summaries[0]['ps']), 'the folds have different SNPs'
    scores = np.array([s['neg_log10_p'][args.trait - 1] for s in summaries])
    correlation = plot_fold_agreement(scores, args.folds, args.trait, args.out_folds)
    off = correlation[~np.eye(len(args.folds), dtype=bool)]
    print(f'trait {args.trait}: Pearson r of -log10 p between folds: median {np.median(off):.3f}, '
          f'min {off.min():.3f}, max {off.max():.3f}')

    summary = summaries[args.folds.index(args.example_fold)]
    score = summary['neg_log10_p']
    threshold = -np.log10(0.05 / score.shape[1])
    hits = (score > threshold).sum(axis=1)
    rows = [t for t in range(len(score)) if hits[t]]
    title = (f'fold {args.example_fold}: traits with significant SNPs' if rows
             else f'fold {args.example_fold}: no trait has significant SNPs')
    rows = rows or list(range(len(score)))
    labels = [f'trait {t + 1}' for t in rows]
    for path in (Path(args.out_manhattan), Path(args.out_manhattan).with_suffix('.png')):
        plot_gwas(path, summary, rows, labels, title)
    for t in range(len(score)):
        print(f'fold {args.example_fold} trait {t + 1}: {hits[t]} significant SNPs, '
              f'lowest p {10 ** -score[t].max():.3g}, lambda GC {summary["lambda_gc"][t]:.3f}')


if __name__ == '__main__':
    main()
