"""The simulation figures of the paper, from the saved results.

--out-figure (paper Fig. 3, simFig): one row per simulation (--row), each with four panels: test
heritability of the true traits and of synthetic traits 1-5; |r| of true trait t with synthetic
trait t; mean ROC curves of trait 1, each with a band from the 25th to the 75th percentile of the
curves of the instances; AUC of traits 1-3. A panel has the same y-axis in all rows. The letters
go down the columns (a, b in column 1).
--out-compare (paper Fig. S2, simCompareSNP): mean ROC curves of traits 2 and 3 of --compare-roc,
with the same bands.
--out-example (simGWASexample): QQ and Manhattan plots of the GWAS of trait 1 of one method on one
instance (--gwas-example).
Each figure is also saved as a .png next to its .pdf. The panels use the functions of
plot_traits.py and plot_gwas.py, which make the same panels as separate files.

The figures show the ground truth, one H2-opt variant (--h2opt) and the baselines (--baselines).
The H2-opt variant gets the label and the color of H2-opt.

    python pipelines/scripts/simulation/plot_paper.py \\
        --row Hyperspectral simulation/evaluation.npz simulation/roc.npz \\
        --row Images helios_simulation/evaluation.npz helios_simulation/roc.npz \\
        --h2opt h2opt_hybrid --baselines pca genetic_pca pch max_features \\
        --compare-roc simulation/roc.npz --gwas-example instance0/gwas/h2opt_hybrid.npz \\
        --out-figure figures/paper/simFig.pdf --out-compare figures/paper/simCompareSNP.pdf \\
        --out-example figures/paper/simGWASexample.pdf
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


def select_methods(result, axes, shown, h2opt):
    """A copy of an evaluate.py or roc.py result with only the methods in shown.

    axes: the per-method arrays to subset, as {key: method axis}. The H2-opt variant h2opt takes
    the slug 'h2opt', so that the figures give it the label and the color of H2-opt."""
    methods = list(result['methods'])
    index = [methods.index(s) for s in shown]
    selected = {k: result[k] for k in result.files}
    for key, axis in axes.items():
        selected[key] = np.take(result[key], index, axis=axis)
    selected['methods'] = np.array(['h2opt' if s == h2opt else s for s in shown])
    return selected


def roc_bands(ax, roc, t):
    """Band from the 25th to the 75th percentile over the instances of the ROC curves of trait t
    (0-based) of each method, in the color of the method."""
    for j, m in enumerate(roc['methods']):
        low, high = np.percentile(roc['tpr_instances'][t, j], [25, 75], axis=0)
        ax.fill_between(roc['fpr'], low, high, color=COLORS[m], alpha=0.15, linewidth=0)


def letter(ax, text):
    ax.text(-0.3, 1.04, text, transform=ax.transAxes, fontsize=11, fontweight='bold',
            va='bottom')


def share_y(axes):
    """Give the axes the union of their y ranges."""
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(low, high)


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def simulation_row(axes, ev, roc, last):
    """The four panels of one simulation; the x labels only in the last row. Returns the slugs of
    the heritability panel (all methods shown)."""
    slugs, values, n_traits = heritability_values(ev)
    boxplot_panel(axes[0], n_traits, slugs, values, 'test heritability')
    axes[0].axhline(0, color='0.3', linewidth=1, linestyle=':')
    cor_slugs, values = correlation_values(ev)
    boxplot_panel(axes[1], len(next(iter(values.values()))), cor_slugs, values,
                  '|r| with true trait')
    roc_bands(axes[2], roc, 0)
    plot_roc(axes[2], roc, 0)
    axes[2].get_legend().remove()
    axes[2].set_ylabel('true positive rate (trait 1)')
    auc_slugs, values = auc_values(roc)
    boxplot_panel(axes[3], len(next(iter(values.values()))), auc_slugs, values,
                  'AUC of causal 100 kb bins')
    if not last:
        for ax in axes:
            ax.set_xlabel('')
    return slugs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', nargs=3, action='append', required=True,
                        metavar=('LABEL', 'EVALUATION', 'ROC'),
                        help='one row of simFig: label, evaluate.py result, roc.py result')
    parser.add_argument('--h2opt', required=True,
                        help='slug of the H2-opt variant shown as H2-opt')
    parser.add_argument('--baselines', nargs='+', required=True, help='slugs of the baselines')
    parser.add_argument('--compare-roc', required=True, help='roc.py result of simCompareSNP')
    parser.add_argument('--gwas-example', required=True,
                        help='gwas.summarize result of the example (trait 1 is shown)')
    parser.add_argument('--out-figure', required=True, help='simFig .pdf')
    parser.add_argument('--out-compare', required=True, help='simCompareSNP .pdf')
    parser.add_argument('--out-example', required=True, help='simGWASexample .pdf')
    args = parser.parse_args()
    plt.rcParams.update({'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
                         'xtick.labelsize': FONT_SIZE - 1, 'ytick.labelsize': FONT_SIZE - 1})
    shown = [args.h2opt] + args.baselines
    roc_axes = {'auc': 1, 'tpr': 1, 'tpr_instances': 1}

    n_rows = len(args.row)
    fig, axes = plt.subplots(n_rows, 4, figsize=(7.2, 0.4 + 2.3 * n_rows), squeeze=False,
                             gridspec_kw={'width_ratios': [1.7, 1, 1, 1]})
    for r, (label, evaluation, roc) in enumerate(args.row):
        ev = select_methods(np.load(evaluation), {'herit': 0, 'cor': 0}, shown, args.h2opt)
        roc = select_methods(np.load(roc), roc_axes, ['truth'] + shown, args.h2opt)
        slugs = simulation_row(axes[r], ev, roc, r == n_rows - 1)
        for c, ax in enumerate(axes[r]):
            letter(ax, 'abcdefghijklmnop'[n_rows * c + r])
        axes[r, 0].annotate(label, xy=(0, 0.5), xycoords='axes fraction', xytext=(-38, 0),
                            textcoords='offset points', rotation=90, ha='right', va='center',
                            fontsize=FONT_SIZE + 1, fontweight='bold')
    for c in range(4):
        share_y(axes[:, c])
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in slugs]
    fig.legend(handles, [LABELS[s] for s in slugs], loc='upper center', ncol=len(slugs),
               frameon=False, fontsize=FONT_SIZE - 1, handlelength=1.2, columnspacing=1)
    fig.subplots_adjust(left=0.12, right=0.99, top=1 - 0.45 / fig.get_figheight(), bottom=0.09,
                        wspace=0.55, hspace=0.3)
    save(fig, args.out_figure)

    roc = select_methods(np.load(args.compare_roc), roc_axes, ['truth'] + shown, args.h2opt)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for t, ax in zip((1, 2), axes, strict=True):
        roc_bands(ax, roc, t)
        plot_roc(ax, roc, t)
        ax.legend(loc='lower right', fontsize=FONT_SIZE - 1, frameon=True, framealpha=0.9,
                  edgecolor='none')
        ax.set_title(f'trait {t + 1}', fontsize=FONT_SIZE + 1)
    fig.tight_layout()
    save(fig, args.out_compare)

    summary = np.load(args.gwas_example)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={'width_ratios': [1, 2]})
    qq(axes[0], summary, 0)
    manhattan(axes[1], summary, 0)
    for ax, text in zip(axes, 'ab', strict=True):
        letter(ax, text)
    fig.tight_layout()
    save(fig, args.out_example)


if __name__ == '__main__':
    main()
