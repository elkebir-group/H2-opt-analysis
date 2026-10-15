"""Shared plotting helpers: method colors, the style of the paper figures, panel letters, grouped
boxplots, median bands, Manhattan and QQ plots, GWAS per fold and cross-fitted."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from .methods import LABELS, ordered

COLORS = {'truth': 'tab:red', 'h2opt': 'tab:blue',
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


# GWAS per fold and cross-fitted (sorGWAS, miscGWAS): row height in inches, locus label size and
# the points between stacked locus labels.
GWAS_ROW_HEIGHT = 0.46
GWAS_LABEL_SIZE = 5.5
GWAS_LABEL_STEP = 6
# Margins in inches above the panels and below them (x labels).
GWAS_TOP_MARGIN = 0.1
GWAS_BOTTOM_MARGIN = 0.34
# Largest figure height in inches, so that the figure and its caption fit on a page; more rows
# get a smaller row height.
GWAS_MAX_HEIGHT = 7.0
# A locus label left of a mark closer than this part of the x axis to its left end goes to the
# right of the mark.
GWAS_LEFT_EDGE = 0.12


def manhattan_x(chrom, ps, order=None):
    """Position of each SNP on one axis with the chromosomes in order (default: sorted), the
    order, the center of each chromosome and the axis length."""
    order = np.unique(chrom) if order is None else np.asarray(order)
    offset = np.zeros(len(ps), dtype=np.int64)
    centers, start = [], 0
    for c in order:
        on = chrom == c
        offset[on] = start
        centers.append(start + (ps[on].max() + 1) / 2)
        start += ps[on].max() + 1
    return ps + offset, order, centers, start


def gwas_fold_rows(path, chrom, ps, score, lambdas, crossfit, crossfit_lambdas, loci, labels,
                   group_names, tick_names, ymax):
    """The GWAS of each group (a trait or a date) with a significant SNP, per fold and
    cross-fitted, under SIM_FIG_RC and MATH_RC. Saves path (.pdf) and a .png next to it; returns
    the rows as (group, fold or None for the cross-fit). The figure is at most GWAS_MAX_HEIGHT
    tall.

    score: folds x groups x SNPs of -log10 p, lambdas: folds x groups; crossfit: groups x SNPs,
    crossfit_lambdas: groups. loci: gwas.clump of score.reshape(-1, SNPs) (row f * groups + g),
    labels: one per locus. tick_names: {chromosome: x tick label}, in the order of the x axis.
    For each group with a Bonferroni hit (0.05 / SNPs) in some fold, one row per such fold and
    then the cross-fit row (bold label). Each row has a QQ plot (left, with lambda GC) and a
    Manhattan plot (right; dotted line: the threshold). In a fold row, a circle marks the
    strongest significant SNP of each locus that the group hits in that fold; in the cross-fit
    row, the SNP of each of the group's loci with the largest cross-fitted -log10 p. Folds are
    numbered from 1, as in the paper. Panel letters go to the first row of each group."""
    n_folds, n_groups, n_snps = score.shape
    threshold = -np.log10(0.05 / n_snps)
    panels = []
    for g in range(n_groups):
        hit_folds = [f for f in range(n_folds) if (score[f, g] > threshold).any()]
        if hit_folds:
            panels += [(g, f) for f in hit_folds] + [(g, None)]
    x, order, centers, length = manhattan_x(chrom, ps, list(tick_names))
    position = {c: k for k, c in enumerate(order)}
    parity = np.array([position[c] for c in chrom]) % 2

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        height = min(GWAS_ROW_HEIGHT * len(panels) + 0.35, GWAS_MAX_HEIGHT)
        # Rows shorter than GWAS_ROW_HEIGHT: one y label per column and row labels horizontal.
        compact = height < GWAS_ROW_HEIGHT * len(panels) + 0.35
        fig, axes = plt.subplots(len(panels), 2, figsize=(6.5, height),
                                 gridspec_kw={'width_ratios': [1, 4]}, squeeze=False)
        lettered = []
        for i, ((g, f), (qq, ax)) in enumerate(zip(panels, axes, strict=True)):
            last = i == len(panels) - 1
            if f is None:
                s, lam = crossfit[g], crossfit_lambdas[g]
            else:
                s, lam = score[f, g], lambdas[f, g]
            ax.scatter(x, s, s=1, c=np.where(parity, 'tab:red', 'tab:blue'), rasterized=True)
            ax.axhline(threshold, color='black', linestyle=':', linewidth=0.8)
            marks = []
            for locus, label in zip(loci, labels, strict=True):
                group_rows = [r for r in locus['rows'] if r % n_groups == g]
                if not group_rows or (f is not None and f * n_groups + g not in group_rows):
                    continue
                members = locus['snps']
                if f is not None:
                    members = members[s[members] > threshold]
                marks.append((members[np.argmax(s[members])], label))
            # labels of marks close together are stacked downward, the highest mark first
            placed = []
            for j, label in sorted(marks, key=lambda m: -s[m[0]]):
                ax.scatter(x[j], s[j], s=14, facecolors='none', edgecolors='black',
                           linewidths=0.6)
                shift = -GWAS_LABEL_STEP * sum(abs(x[j] - xp) < 0.03 * length for xp in placed)
                placed.append(x[j])
                right = x[j] < GWAS_LEFT_EDGE * length
                ax.annotate(label, (x[j], s[j]), xytext=(4 if right else -4, shift),
                            textcoords='offset points', ha='left' if right else 'right',
                            va='center', fontsize=GWAS_LABEL_SIZE,
                            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.3,
                                  'alpha': 0.8})
            ax.set_xticks(centers, [tick_names[c] for c in order] if last else [])
            ax.tick_params(axis='x', length=0)
            ax.set_xlim(0, length)
            ax.set_ylim(0, ymax)
            ax.spines[['top', 'right']].set_visible(False)
            name = 'cross-fit' if f is None else f'fold {f + 1}'
            ax.text(1.01, 0.5, f'{group_names[g]}\n{name}', transform=ax.transAxes,
                    rotation=0 if compact else 270, va='center', ha='left', fontsize=6.5,
                    fontweight='bold' if f is None else 'normal')

            observed = np.sort(s)
            expected = -np.log10(np.arange(len(observed), 0, -1) / (len(observed) + 1))
            qq.scatter(expected, observed, s=2, color='tab:blue', rasterized=True)
            top = min(expected.max(), observed.max())
            qq.plot([0, top], [0, top], color='black', linewidth=0.6, linestyle='--', zorder=1)
            qq.text(0.04, 0.96, rf'$\lambda_\mathrm{{GC}}$ = {lam:.3f}', transform=qq.transAxes,
                    va='top', ha='left')
            qq.set_xlim(0, None)
            qq.set_ylim(0, ymax)
            if not compact:
                ax.set_ylabel(r'$-\log_{10} p$')
                qq.set_ylabel(r'$-\log_{10} p$')
            qq.spines[['top', 'right']].set_visible(False)
            if not last:
                qq.set_xticklabels([])
            if g not in lettered:
                lettered.append(g)
                letter(qq, 'abcdefghijklmnop'[len(lettered) - 1], -0.45)
        axes[-1, 1].set_xlabel('chromosome')
        axes[-1, 0].set_xlabel(r'expected $-\log_{10} p$')
        # x tick labels of two lines need more room below
        bottom = GWAS_BOTTOM_MARGIN + 0.1 * max(t.count('\n') for t in tick_names.values())
        fig.subplots_adjust(left=0.07, right=0.92 if compact else 0.95,
                            top=1 - GWAS_TOP_MARGIN / height, bottom=bottom / height,
                            hspace=0.3, wspace=0.25)
        if compact:
            for column in (0, 1):
                box = axes[0, column].get_position()
                fig.text(box.x0 - 0.045, 0.5, r'$-\log_{10} p$', rotation=90, va='center',
                         ha='right')
        fig.savefig(path)
        fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=200)
        plt.close(fig)
    return panels
