"""The simulation figures of the paper, from the saved results.

--out-figure (paper Fig. 3, simFig): a test heritability of the true traits and of synthetic
traits 1-5; b |r| of true trait t with synthetic trait t; c QQ and Manhattan plots of the GWAS of
trait 1 of one method on one instance; d mean ROC curves of trait 1; e AUC of traits 1-3.
--out-compare (paper Fig. S2, simCompareSNP): mean ROC curves of traits 2 and 3. Each figure is
also saved as a .png next to its .pdf. The panels use the functions of plot_traits.py and
plot_gwas.py, which make the same panels as separate files.

    python pipelines/scripts/simulation/plot_paper.py --evaluation evaluation.npz \\
        --roc roc.npz --gwas-example instance0/gwas/h2opt.npz \\
        --example-label "H2-opt, instance 0, trait 1" \\
        --out-figure figures/paper/simFig.pdf --out-compare figures/paper/simCompareSNP.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_gwas import auc_values, plot_roc  # noqa: E402
from plot_traits import correlation_values, heritability_values  # noqa: E402

from h2opt_analysis.figures import COLORS, grouped_boxplot  # noqa: E402
from h2opt_analysis.methods import LABELS  # noqa: E402

FONT_SIZE = 8


def manhattan(ax, summary, row):
    """Manhattan plot of one run of a gwas.summarize result; chromosomes alternate red and blue,
    the dotted line is the Bonferroni threshold 0.05 / SNPs."""
    chromosome, position, score = summary['chr'], summary['ps'], summary['neg_log10_p'][row]
    order = np.unique(chromosome)
    offset = np.zeros(len(position), dtype=np.int64)
    start = 0
    for c in order:
        on = chromosome == c
        offset[on] = start
        start += position[on].max() + 1
    parity = np.searchsorted(order, chromosome) % 2
    ax.scatter(position + offset, score, s=1, c=np.where(parity, 'tab:red', 'tab:blue'),
               rasterized=True)
    ax.axhline(-np.log10(0.05 / len(score)), color='black', linestyle=':', linewidth=1)
    ax.set_xticks([])
    ax.set_xlim(0, start)
    ax.set_xlabel('genome position')
    ax.set_ylabel('-log10 p')
    ax.spines[['top', 'right']].set_visible(False)


def qq(ax, summary, row):
    """QQ plot of one run of a gwas.summarize result, with its genomic-control lambda."""
    observed = np.sort(summary['neg_log10_p'][row])
    expected = -np.log10(np.arange(len(observed), 0, -1) / (len(observed) + 1))
    ax.scatter(expected, observed, s=2, color='tab:blue', rasterized=True)
    top = min(expected.max(), observed.max())
    ax.plot([0, top], [0, top], 'r--', linewidth=1)
    ax.text(0.03, 0.95, f'lambda GC {summary["lambda_gc"][row]:.3f}', transform=ax.transAxes,
            va='top', fontsize=FONT_SIZE - 1)
    ax.set_xlabel('expected -log10 p')
    ax.set_ylabel('observed -log10 p')
    ax.spines[['top', 'right']].set_visible(False)


def boxplot_panel(ax, n_traits, slugs, values, ylabel):
    grouped_boxplot(ax, [str(t + 1) for t in range(n_traits)], slugs, values)
    ax.set_xlabel('trait')
    ax.set_ylabel(ylabel)


def letter(ax, text, x=-0.14):
    ax.text(x, 1.02, text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--roc', required=True)
    parser.add_argument('--gwas-example', required=True,
                        help='gwas.summarize result of the example (trait 1 is shown)')
    parser.add_argument('--example-label', required=True)
    parser.add_argument('--out-figure', required=True, help='simFig .pdf')
    parser.add_argument('--out-compare', required=True, help='simCompareSNP .pdf')
    args = parser.parse_args()
    plt.rcParams.update({'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
                         'xtick.labelsize': FONT_SIZE - 1, 'ytick.labelsize': FONT_SIZE - 1})
    ev, roc = np.load(args.evaluation), np.load(args.roc)
    summary = np.load(args.gwas_example)

    fig = plt.figure(figsize=(7.2, 5.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.45)
    top = grid[0].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)
    bottom = grid[1].subgridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.45)

    ax = fig.add_subplot(top[0])
    slugs, values, n_traits = heritability_values(ev)
    boxplot_panel(ax, n_traits, slugs, values, 'test heritability')
    ax.axhline(0, color='0.3', linewidth=1, linestyle=':')
    letter(ax, 'a', -0.1)
    all_slugs = slugs

    ax = fig.add_subplot(top[1])
    slugs, values = correlation_values(ev)
    boxplot_panel(ax, len(next(iter(values.values()))), slugs, values, '|r| with true trait')
    letter(ax, 'b')

    c = bottom[0].subgridspec(2, 1, hspace=0.65)
    ax = fig.add_subplot(c[0])
    qq(ax, summary, 0)
    ax.set_title(args.example_label, fontsize=FONT_SIZE)
    letter(ax, 'c', -0.3)
    manhattan(fig.add_subplot(c[1]), summary, 0)

    ax = fig.add_subplot(bottom[1])
    plot_roc(ax, roc, 0)
    ax.legend(loc='lower right', fontsize=FONT_SIZE - 2.5, frameon=True, framealpha=0.9,
              edgecolor='none', borderpad=0.2, handlelength=1.2)
    ax.set_title('trait 1, mean over instances', fontsize=FONT_SIZE)
    letter(ax, 'd', -0.22)

    ax = fig.add_subplot(bottom[2])
    slugs, values = auc_values(roc)
    boxplot_panel(ax, len(next(iter(values.values()))), slugs, values, 'AUC of causal 100 kb bins')
    letter(ax, 'e', -0.24)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in all_slugs]
    fig.legend(handles, [LABELS[s] for s in all_slugs], loc='upper center', ncol=len(all_slugs),
               frameon=False, fontsize=FONT_SIZE - 1, handlelength=1.2, columnspacing=1)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.08)
    save(fig, args.out_figure)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for t, ax in zip((1, 2), axes, strict=True):
        plot_roc(ax, roc, t)
        ax.legend(loc='lower right', fontsize=FONT_SIZE - 1, frameon=True, framealpha=0.9,
                  edgecolor='none')
        ax.set_title(f'trait {t + 1}', fontsize=FONT_SIZE + 1)
    fig.tight_layout()
    save(fig, args.out_compare)


if __name__ == '__main__':
    main()
