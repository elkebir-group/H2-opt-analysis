"""The simulation figures of the paper, from the saved results.

--out-figure (paper Fig. 3, simFig): one row per simulation (--row), each with four panels: test
heritability of the true traits and of synthetic traits 1-5; |r| of true trait t with synthetic
trait t; mean ROC curves of trait 1, each with a band from the 25th to the 75th percentile of the
curves of the instances and the median AUC over the instances in the legend; AUC of traits 1-3. A
panel has the same y-axis in all rows. The letters go down the columns (a, b in column 1). The
row labels are at the right of the rows.
--out-compare (paper Fig. S2, simCompareSNP): mean ROC curves of traits 2 and 3 of --compare-roc,
with the same bands (lighter).
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
from matplotlib.ticker import MaxNLocator  # noqa: E402
from plot_gwas import auc_values, plot_roc  # noqa: E402
from plot_traits import correlation_values, heritability_values  # noqa: E402

from h2opt_analysis.figures import COLORS  # noqa: E402
from h2opt_analysis.methods import LABELS  # noqa: E402

# simFig is drawn at the text width of the paper (6.5 in), so that LaTeX does not scale it. Its
# text is Liberation Sans (the metrics of Arial): 6.5 pt labels and legend, 6 pt ticks.
# simCompareSNP and simGWASexample keep their earlier sizes and font sizes (7 to 11 pt).
FONT_SIZE = 6.5
TICK_SIZE = 6
ROC_LEGEND_SIZE = 5.5
ROW_LABEL_SIZE = 7.5
LETTER_SIZE = 8.5
SIM_FIG_SIZE = (6.5, 3.7)
SIM_FIG_RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans'],
              'pdf.fonttype': 42, 'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
              'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE,
              'legend.fontsize': FONT_SIZE, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
              'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
              'lines.linewidth': 1.0}
# simFig columns: heritability (traits 1-5), |r| (traits 1-3), ROC, AUC (traits 1-3).
WIDTH_RATIOS = [4.6, 3, 3.6, 3]
# x of the panel letters, in axes coordinates of each column.
LETTER_X = [-0.21, -0.33, -0.28, -0.37]
# simFig margins in inches: above the panels (legend and letters) and below them (x labels);
# LEGEND_GAP is between the top of the panels and the bottom of the legend.
TOP_MARGIN = 0.31
BOTTOM_MARGIN = 0.32
LEGEND_GAP = 0.11
# Part of the unit between two traits that the boxes of one trait span.
GROUP_SPAN = 0.72


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
            va='top', fontsize=7)
    ax.set_xlabel('expected -log10 p')
    ax.set_ylabel('observed -log10 p')
    ax.spines[['top', 'right']].set_visible(False)


def boxplot_panel(ax, n_traits, slugs, values, ylabel):
    """Boxplots of values[slug][trait] side by side within each trait. The boxes of one trait span
    GROUP_SPAN of the unit between traits, which leaves a clear gap between traits."""
    step = GROUP_SPAN / len(slugs)
    for s, slug in enumerate(slugs):
        keep = [t for t in range(n_traits) if len(values[slug].get(t, []))]
        boxes = ax.boxplot([values[slug][t] for t in keep],
                           positions=[t + (s - (len(slugs) - 1) / 2) * step for t in keep],
                           widths=step * 0.75, patch_artist=True, showfliers=False,
                           medianprops={'color': 'black', 'linewidth': 0.8},
                           whiskerprops={'linewidth': 0.7}, capprops={'linewidth': 0.7},
                           boxprops={'linewidth': 0.5})
        for box in boxes['boxes']:
            box.set(facecolor=COLORS[slug], edgecolor=COLORS[slug], alpha=0.85)
        for line in boxes['whiskers'] + boxes['caps']:
            line.set_color(COLORS[slug])
    ax.set_xticks(range(n_traits), [str(t + 1) for t in range(n_traits)])
    ax.set_xlim(-0.5, n_traits - 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
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


def roc_bands(ax, roc, t, alpha):
    """Band from the 25th to the 75th percentile over the instances of the ROC curves of trait t
    (0-based) of each method, in the color of the method."""
    for j, m in enumerate(roc['methods']):
        low, high = np.percentile(roc['tpr_instances'][t, j], [25, 75], axis=0)
        ax.fill_between(roc['fpr'], low, high, color=COLORS[m], alpha=alpha, linewidth=0)


def letter(ax, text, x):
    ax.text(x, 1.0, text, transform=ax.transAxes, fontsize=LETTER_SIZE, fontweight='bold',
            va='bottom', ha='left')


def row_label(fig, ax, text):
    """Label of a row of panels, rotated, just to the right of the last panel ax of the row."""
    box = ax.get_position()
    fig.text(box.x1 + 0.04 / fig.get_figwidth(), (box.y0 + box.y1) / 2, text, rotation=-90,
             ha='left', va='center', fontsize=ROW_LABEL_SIZE, fontweight='bold')


def share_y(axes):
    """Give the axes the union of their y ranges."""
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(low, high)


def top_legend(fig, slugs, top):
    """One legend for the figure, in one line above the panels; top is the top of the panels in
    figure coordinates."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in slugs]
    fig.legend(handles, [LABELS[s] for s in slugs], loc='lower center',
               bbox_to_anchor=(0.5, top + LEGEND_GAP / fig.get_figheight()), borderaxespad=0,
               ncol=len(slugs), frameon=False,
               handlelength=1.2, handletextpad=0.5, columnspacing=1.4)


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def simulation_row(axes, ev, roc):
    """The four panels of one simulation. Returns the slugs of the heritability panel (all
    methods shown)."""
    slugs, values, n_traits = heritability_values(ev)
    boxplot_panel(axes[0], n_traits, slugs, values, 'test heritability')
    axes[0].axhline(0, color='0.3', linewidth=0.8, linestyle=':')
    cor_slugs, values = correlation_values(ev)
    boxplot_panel(axes[1], len(next(iter(values.values()))), cor_slugs, values,
                  '|r| with true trait')
    roc_bands(axes[2], roc, 0, 0.25)
    plot_roc(axes[2], roc, 0)
    axes[2].legend(loc='lower right', fontsize=ROC_LEGEND_SIZE, frameon=True, framealpha=0.8,
                   edgecolor='none', borderpad=0.2, borderaxespad=0.2, handlelength=1,
                   handletextpad=0.4, labelspacing=0.15)
    axes[2].set_ylabel('true positive rate')
    axes[2].set_xlabel('false positive rate')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
    axes[2].set_yticks([0, 0.5, 1], ['0', '0.5', '1'])
    auc_slugs, values = auc_values(roc)
    boxplot_panel(axes[3], len(next(iter(values.values()))), auc_slugs, values,
                  'AUC of causal 100 kb bins')
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
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7})
    shown = [args.h2opt] + args.baselines
    roc_axes = {'auc': 1, 'tpr': 1, 'tpr_instances': 1}
    letters = 'abcdefghijklmnop'
    n_rows = len(args.row)

    with plt.rc_context(SIM_FIG_RC):
        fig, axes = plt.subplots(n_rows, 4, figsize=SIM_FIG_SIZE, squeeze=False,
                                 gridspec_kw={'width_ratios': WIDTH_RATIOS})
        for r, (_, evaluation, roc) in enumerate(args.row):
            ev = select_methods(np.load(evaluation), {'herit': 0, 'cor': 0}, shown, args.h2opt)
            roc = select_methods(np.load(roc), roc_axes, ['truth'] + shown, args.h2opt)
            slugs = simulation_row(axes[r], ev, roc)
        for c in range(4):
            share_y(axes[:, c])
            if c != 2:
                for ax in axes[:, c]:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            for r in range(n_rows):
                letter(axes[r, c], letters[n_rows * c + r], LETTER_X[c])
                if r < n_rows - 1:
                    axes[r, c].set_xlabel('')
        top = 1 - TOP_MARGIN / SIM_FIG_SIZE[1]
        fig.subplots_adjust(left=0.055, right=0.975, top=top,
                            bottom=BOTTOM_MARGIN / SIM_FIG_SIZE[1], wspace=0.35, hspace=0.27)
        top_legend(fig, slugs, top)
        for r, (label, _, _) in enumerate(args.row):
            row_label(fig, axes[r, -1], label)
        save(fig, args.out_figure)

    roc = select_methods(np.load(args.compare_roc), roc_axes, ['truth'] + shown, args.h2opt)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for t, ax in zip((1, 2), axes, strict=True):
        roc_bands(ax, roc, t, 0.15)
        plot_roc(ax, roc, t)
        ax.legend(loc='lower right', fontsize=7, frameon=True, framealpha=0.9,
                  edgecolor='none')
        ax.set_title(f'trait {t + 1}', fontsize=9)
    fig.tight_layout()
    save(fig, args.out_compare)

    summary = np.load(args.gwas_example)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={'width_ratios': [1, 2]})
    qq(axes[0], summary, 0)
    manhattan(axes[1], summary, 0)
    for ax, text in zip(axes, 'ab', strict=True):
        ax.text(-0.3, 1.04, text, transform=ax.transAxes, fontsize=11, fontweight='bold',
                va='bottom')
    fig.tight_layout()
    save(fig, args.out_example)


if __name__ == '__main__':
    main()
