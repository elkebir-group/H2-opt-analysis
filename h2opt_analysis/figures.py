"""Shared plotting helpers: method colors, the style of the paper figures, panel letters, grouped
boxplots, median bands, Manhattan and QQ plots."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from .methods import LABELS, ordered

COLORS = {'truth': 'tab:red', 'h2opt_paper': 'tab:blue', 'h2opt': 'tab:blue',
          'h2opt_hybrid': 'tab:cyan',
          'pca': 'tab:orange', 'genetic_pca': 'tab:purple',
          'pch': 'tab:pink', 'max_features': 'tab:green', 'single_band': 'tab:olive'}


# Style of the paper figures, drawn at their size in the paper so that LaTeX does not scale them:
# Liberation Sans (the metrics of Arial), 6.5 pt labels and legend, 6 pt ticks, 8.5 pt bold panel
# letters.
FONT_SIZE = 6.5
TICK_SIZE = 6
LETTER_SIZE = 8.5
SIM_FIG_RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans'],
              'pdf.fonttype': 42, 'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
              'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE,
              'legend.fontsize': FONT_SIZE, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
              'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
              'lines.linewidth': 1.0}
# Math labels in the text font of the figure.
MATH_RC = {'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
           'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold'}


def letter(ax, text, x):
    """Panel letter at the top left of ax; x is in axes coordinates."""
    ax.text(x, 1.0, text, transform=ax.transAxes, fontsize=LETTER_SIZE, fontweight='bold',
            va='bottom', ha='left')


def grouped_boxplot(ax, groups, slugs, values):
    """Boxplots of values[slug][group] (a list per group), side by side within each group."""
    width = 0.8 / len(slugs)
    for s, slug in enumerate(slugs):
        positions = [g + (s - (len(slugs) - 1) / 2) * width for g in range(len(groups))]
        data = [values[slug].get(g, []) for g in range(len(groups))]
        keep = [k for k, d in enumerate(data) if len(d)]
        boxes = ax.boxplot([data[k] for k in keep], positions=[positions[k] for k in keep],
                           widths=width * 0.9, patch_artist=True, showfliers=False,
                           medianprops={'color': 'black'})
        for box in boxes['boxes']:
            box.set(facecolor=COLORS[slug], edgecolor=COLORS[slug], alpha=0.8)
    ax.set_xticks(range(len(groups)), groups)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.9')
    ax.set_axisbelow(True)


def median_bands(ax, x, values, slugs):
    """Per method, a line through the median over the folds and a band from the 25th to the 75th
    percentile. values: (slugs, folds, len(x)). The methods are drawn in the reverse figure order,
    so the first method of the figure order (H2-opt) is on top."""
    slugs = list(slugs)
    for slug in ordered(slugs)[::-1]:
        low, median, high = np.percentile(values[slugs.index(slug)], [25, 50, 75], axis=0)
        ax.fill_between(x, low, high, color=COLORS[slug], alpha=0.25, linewidth=0)
        ax.plot(x, median, 'o-', color=COLORS[slug], ms=1.5, lw=0.8, label=LABELS[slug])


def ordered_legend(ax, labels, **kwargs):
    """Legend of the artists of ax with the given labels, in the order of labels."""
    handles, names = ax.get_legend_handles_labels()
    keep = [names.index(label) for label in labels]
    ax.legend([handles[i] for i in keep], [names[i] for i in keep], **kwargs)


def legend(fig, slugs):
    """Figure legend with one colored patch per method."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in slugs]
    fig.legend(handles, [LABELS[s] for s in slugs], loc='upper center',
               ncol=min(len(slugs), 4), frameon=False, fontsize=8)


def save_trait_boxplots(path, n_traits, slugs, values, ylabel, width=5):
    """Figure of grouped boxplots per trait (values[slug][trait] is a list over instances)."""
    fig, ax = plt.subplots(figsize=(width, 3.2))
    grouped_boxplot(ax, [str(t + 1) for t in range(n_traits)], slugs, values)
    ax.set_xlabel('trait')
    ax.set_ylabel(ylabel)
    legend(fig, slugs)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(path)
    plt.close(fig)


def plot_noise_choices(levels, chosen, scores, path, series='trait', series_labels=None):
    """Noise levels chosen by h2opt.selection.select_noise_level in several subsets (folds).

    chosen: (folds,) levels; scores: (folds, series, splits, levels), the series being traits or
    measurement sets (e.g. dates; series names them, series_labels gives their labels, default
    1, 2, ...). Left: the held-out heritability averaged over the series and splits per level, one
    line per fold, its chosen level marked. Right: the held-out heritability of each series
    averaged over the folds and splits per level. Saves path (.pdf) and a .png next to it."""
    levels = np.asarray(levels)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for fold, mean in enumerate(scores.mean(axis=(1, 2))):
        left.plot(levels, mean, '-', color='0.6', lw=1)
        a = int(np.argmin(np.abs(levels - chosen[fold])))
        left.plot(levels[a], mean[a], 'o', color='tab:blue', ms=4)
    left.set_ylabel(f'held-out heritability, mean of {series}s')
    left.set_title('one line per fold; dot: chosen', fontsize=9)
    n_series = scores.shape[1]
    labels = series_labels if series_labels is not None else range(1, n_series + 1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_series))
    for t, label in zip(range(n_series), labels, strict=True):
        right.plot(levels, scores[:, t].mean(axis=(0, 1)), '-', color=colors[t], lw=1,
                   label=f'{label}')
    right.set_ylabel('held-out heritability')
    right.set_title('mean over folds', fontsize=9)
    right.legend(title=series, frameon=False, fontsize=6, title_fontsize=7,
                 ncol=2 if n_series > 12 else 1, loc='center left', bbox_to_anchor=(1, 0.5))
    for ax in (left, right):
        ax.set_xscale('log')
        ax.set_xlabel('noise level (noise SD / noise_scale)')
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)


def plot_gwas(path, summary, rows, row_labels, title, dpi=150):
    """Manhattan (left) and QQ (right) plots of rows of a gwas.summarize result, one row each.

    summary: dict or .npz with chr, ps, neg_log10_p (runs x SNPs) and lambda_gc. rows: indices
    into its runs; row_labels: their y labels. The dotted line is the Bonferroni threshold
    0.05 / SNPs. Chromosomes alternate red and blue in their sorted order."""
    chromosome, position, score = summary['chr'], summary['ps'], summary['neg_log10_p']
    offset = np.zeros(len(position), dtype=np.int64)
    start = 0
    for c in np.unique(chromosome):
        on = chromosome == c
        offset[on] = start
        start += position[on].max() + 1
    parity = np.searchsorted(np.unique(chromosome), chromosome) % 2
    threshold = -np.log10(0.05 / score.shape[1])
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 1.9 * len(rows)), squeeze=False,
                             gridspec_kw={'width_ratios': [3, 1]})
    for ax_row, r, label in zip(axes, rows, row_labels, strict=True):
        ax = ax_row[0]
        ax.scatter(position + offset, score[r], s=1, c=np.where(parity, 'tab:red', 'tab:blue'),
                   rasterized=True)
        ax.axhline(threshold, color='black', linestyle=':', linewidth=1)
        ax.set_ylabel(f'{label}\n-log10 p')
        ax.set_xticks([])
        ax.spines[['top', 'right']].set_visible(False)
        ax = ax_row[1]
        observed = np.sort(score[r])
        expected = -np.log10(np.arange(len(observed), 0, -1) / (len(observed) + 1))
        ax.scatter(expected, observed, s=1, rasterized=True)
        top = min(expected.max(), observed.max())
        ax.plot([0, top], [0, top], 'r--', linewidth=1)
        ax.set_title(f'lambda GC {summary["lambda_gc"][r]:.3f}', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1, 0].set_xlabel('genome position')
    axes[-1, 1].set_xlabel('expected -log10 p')
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# Real results against the permutation null (simNull and the null figures of the real datasets):
# the null in gray, the real results in the color of H2-opt, one marked result (the cross-fit) as
# a black diamond.
NULL_COLOR = '0.55'
NULL_LINE_COLOR = '0.78'
NULL_FILL = '0.92'
REAL_POINT_COLOR = '#0b3c66'  # the points over the H2-opt boxes
MARK_COLOR = 'black'


def real_vs_null(ax, real, null, mark=None):
    """Per trait (column): the real values (rows of real) as a box in the color of H2-opt with
    dark points (one row: a diamond in the color of H2-opt), and the null values (rows of null)
    as a light box with gray points beside it. mark: None, or one value per trait (NaN: none)
    drawn as a black diamond on the real box."""
    rng = np.random.default_rng(0)
    n_traits = real.shape[1]
    for t in range(n_traits):
        if len(real) == 1:
            ax.scatter([t - 0.2], real[:, t], s=22, marker='D', color=COLORS['h2opt'],
                       edgecolor='white', linewidth=0.4, zorder=4)
            continue
        boxes = ax.boxplot([real[:, t]], positions=[t - 0.2], widths=0.32, showfliers=False,
                           patch_artist=True, medianprops={'color': 'black', 'linewidth': 0.8},
                           whiskerprops={'color': COLORS['h2opt'], 'linewidth': 0.7},
                           capprops={'color': COLORS['h2opt'], 'linewidth': 0.7},
                           boxprops={'linewidth': 0.5})
        boxes['boxes'][0].set(facecolor=COLORS['h2opt'], edgecolor=COLORS['h2opt'], alpha=0.85)
        ax.scatter(t - 0.2 + rng.uniform(-0.08, 0.08, len(real)), real[:, t], s=5,
                   color=REAL_POINT_COLOR, linewidth=0, zorder=3)
    for t in range(n_traits):
        ax.boxplot([null[:, t]], positions=[t + 0.2], widths=0.32, showfliers=False,
                   patch_artist=True, boxprops={'facecolor': NULL_FILL, 'edgecolor': NULL_COLOR,
                                                'linewidth': 0.5},
                   medianprops={'color': NULL_COLOR, 'linewidth': 0.8},
                   whiskerprops={'color': NULL_COLOR, 'linewidth': 0.6},
                   capprops={'color': NULL_COLOR, 'linewidth': 0.6})
        ax.scatter(t + 0.2 + rng.uniform(-0.08, 0.08, len(null)), null[:, t], s=5,
                   color=NULL_COLOR, linewidth=0, zorder=3)
    if mark is not None:
        ax.scatter(np.arange(n_traits) - 0.2, mark, s=22, marker='D', color=MARK_COLOR,
                   edgecolor='white', linewidth=0.4, zorder=4)
    ax.set_xticks(range(n_traits), [str(t + 1) for t in range(n_traits)])
    ax.set_xlim(-0.5, n_traits - 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('trait')


def qq_curve(ax, score, color, linewidth, zorder, alpha=1.0):
    """QQ curve of one GWAS (-log10 p of all SNPs); returns the largest expected -log10 p."""
    observed = np.sort(score)
    expected = -np.log10(np.arange(len(observed), 0, -1) / (len(observed) + 1))
    ax.plot(expected, observed, color=color, linewidth=linewidth, zorder=zorder, alpha=alpha,
            rasterized=True)
    return expected.max()


def qq_real_vs_null(ax, reals, nulls, mark=None):
    """QQ plot: each null GWAS in light gray, each real GWAS in the color of H2-opt (thin when
    there are several), mark (one GWAS, or None) in black; the diagonal (dashed) and the
    Bonferroni line 0.05 / SNPs (dotted)."""
    for score in nulls:
        qq_curve(ax, score, NULL_LINE_COLOR, 0.8, 2)
    for score in reals:
        top = qq_curve(ax, score, COLORS['h2opt'], 1.0 if len(reals) == 1 else 0.6, 3,
                       1.0 if len(reals) == 1 else 0.6)
    if mark is not None:
        qq_curve(ax, mark, MARK_COLOR, 1.0, 4)
    ax.plot([0, top], [0, top], color='black', linewidth=0.6, linestyle='--', zorder=1)
    ax.axhline(-np.log10(0.05 / len(reals[0])), color='black', linestyle=':', linewidth=0.8)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel(r'expected $-\log_{10} p$')
