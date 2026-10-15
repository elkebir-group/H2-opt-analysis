"""The Miscanthus GWAS controls of the paper (miscNull), in the style of sorNull: H2-opt (CNN)
trait 1 of the 5 folds on every date, its cross-fitted GWAS and the permutation null, after the
nearest-neighbor field adjustment (the adjusted run). One row per population.

a: per date, the narrow-sense test heritability of the 5 folds (4 x Henderson3, not clipped; a box
in the color of H2-opt (CNN) with a point per fold); at the right, the permutation null
(null_gwas.py on the selection date, fold 0; a gray box with a point per permutation); a line at 0.
b: the same for the largest -log10 p, with the cross-fit (black diamond) and the Bonferroni line
0.05 / SNPs. c: the QQ plot of the date with significant SNPs in the most folds (the first such
date on a tie): each fold, the cross-fit and each permutation. d: as b for the genomic-control
lambda, with a line at 1. Drawn at the text width of the paper (6.5 in), saved as .pdf and,
beside it, .png; the numbers are printed.

    python pipelines/scripts/miscanthus/plot_paper_null.py \\
        --gwas central/gwas/date0.npz ... central/gwas/date13.npz \\
        --crossfit central/gwas/crossfit_date0.npz ... --null central/null/perm*.npz \\
        --heritability-central central/heritability.npz --gwas-south ... --crossfit-south ... \\
        --null-south ... --heritability-south ... --out figures/paper/miscNull.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from h2opt_analysis import miscanthus  # noqa: E402
from h2opt_analysis.figures import (  # noqa: E402
    COLORS,
    MARK_COLOR,
    MATH_RC,
    NULL_COLOR,
    NULL_FILL,
    NULL_LINE_COLOR,
    REAL_POINT_COLOR,
    SIM_FIG_RC,
    TICK_SIZE,
    letter,
    qq_real_vs_null,
)

FIG_SIZE = (6.5, 4.4)
# Narrow-sense heritability of half-sib families: 1 / (1/4) times the Henderson3 estimate.
HALF_SIB = 4
COLOR = COLORS['h2opt_hybrid']
ROW_LABEL_SIZE = 7.5


def boxes(ax, real, mark, null, null_label):
    """Per date (column of real, folds x dates): a box of the folds and the cross-fit mark (one
    value per date); then one gray box of null (one value per permutation) labeled null_label."""
    rng = np.random.default_rng(0)
    n = real.shape[1]
    for d in range(n):
        box = ax.boxplot([real[:, d]], positions=[d], widths=0.6, showfliers=False,
                         patch_artist=True, medianprops={'color': 'black', 'linewidth': 0.8},
                         whiskerprops={'color': COLOR, 'linewidth': 0.7},
                         capprops={'color': COLOR, 'linewidth': 0.7}, boxprops={'linewidth': 0.5})
        box['boxes'][0].set(facecolor=COLOR, edgecolor=COLOR, alpha=0.6)
        ax.scatter(d + rng.uniform(-0.15, 0.15, len(real)), real[:, d], s=4,
                   color=REAL_POINT_COLOR, linewidth=0, zorder=3)
    ax.scatter(np.arange(n), mark, s=16, marker='D', color=MARK_COLOR, edgecolor='white',
               linewidth=0.4, zorder=4)
    ax.boxplot([null], positions=[n + 0.5], widths=0.6, showfliers=False, patch_artist=True,
               boxprops={'facecolor': NULL_FILL, 'edgecolor': NULL_COLOR, 'linewidth': 0.5},
               medianprops={'color': NULL_COLOR, 'linewidth': 0.8},
               whiskerprops={'color': NULL_COLOR, 'linewidth': 0.6},
               capprops={'color': NULL_COLOR, 'linewidth': 0.6})
    ax.scatter(n + 0.5 + rng.uniform(-0.15, 0.15, len(null)), null, s=4, color=NULL_COLOR,
               linewidth=0, zorder=3)
    ax.set_xticks([*range(n), n + 0.5], [*miscanthus.DATES, null_label], rotation=90)
    ax.set_xlim(-0.7, n + 1.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    for population in miscanthus.POPULATIONS:
        parser.add_argument(f'--gwas-{population}', nargs='+', required=True,
                            help='gwas.py result of each date (5 folds), in date order')
        parser.add_argument(f'--crossfit-{population}', nargs='+', required=True,
                            help='gwas.py result of the cross-fitted trait of each date')
        parser.add_argument(f'--null-{population}', nargs='+', required=True,
                            help='null_gwas.py results')
        parser.add_argument(f'--heritability-{population}', required=True,
                            help='evaluate_heritability.py result (method h2opt_hybrid)')
    parser.add_argument('--null-date', type=int, required=True,
                        help='index into miscanthus.DATES of the permutation null')
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    args = parser.parse_args()
    null_label = f'null ({miscanthus.DATES[args.null_date]})'

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig, axes = plt.subplots(2, 4, figsize=FIG_SIZE,
                                 gridspec_kw={'width_ratios': (2.1, 2.1, 1.2, 2.1)})
        for row, population in zip(axes, miscanthus.POPULATIONS, strict=True):
            folds = [np.load(p) for p in getattr(args, f'gwas_{population}')]
            crossfit = [np.load(p) for p in getattr(args, f'crossfit_{population}')]
            nulls = [np.load(p) for p in getattr(args, f'null_{population}')]
            with np.load(getattr(args, f'heritability_{population}')) as f:
                # folds x dates
                h2 = HALF_SIB * f['test'][list(f['methods']).index('h2opt_hybrid')].T
            null_h2 = HALF_SIB * np.array([float(n['h2_test'][0]) for n in nulls])
            n_snps = folds[0]['neg_log10_p'].shape[1]
            threshold = -np.log10(0.05 / n_snps)
            real_max = np.array([f['neg_log10_p'].max(axis=1) for f in folds]).T  # folds x dates
            real_lambda = np.array([f['lambda_gc'] for f in folds]).T
            cross_max = np.array([c['neg_log10_p'].max() for c in crossfit])
            cross_lambda = np.array([float(np.ravel(c['lambda_gc'])[0]) for c in crossfit])
            null_max = np.array([n['neg_log10_p'].max() for n in nulls])
            null_lambda = np.array([float(n['lambda_gc'][0]) for n in nulls])
            hits = (real_max > threshold).sum(axis=0)
            qq_date = int(np.argmax(hits))

            cross_dates = [miscanthus.DATES[d] for d in np.flatnonzero(cross_max > threshold)]
            print(f'{population}: Bonferroni -log10 p {threshold:.2f}; fold-date pairs with a '
                  f'significant SNP {(real_max > threshold).sum()} of {real_max.size}; cross-fit '
                  f'dates with one {cross_dates}; null {(null_max > threshold).sum()} of '
                  f'{len(nulls)} (largest -log10 p {null_max.max():.2f}); lambda GC folds '
                  f'{real_lambda.min():.2f}-{real_lambda.max():.2f}, cross-fit '
                  f'{cross_lambda.min():.2f}-{cross_lambda.max():.2f}, null '
                  f'{null_lambda.min():.2f}-{null_lambda.max():.2f}; QQ date '
                  f'{miscanthus.DATES[qq_date]}; test h2 on the null date: median of the folds '
                  f'{np.median(h2[:, args.null_date]):.3f}, of the null {np.median(null_h2):.3f}')

            a, b, c, d = row
            boxes(a, h2, np.full(h2.shape[1], np.nan), null_h2, null_label)
            a.axhline(0, color='0.3', linewidth=0.8, linestyle=':')
            a.set_ylabel('test heritability')
            boxes(b, real_max, cross_max, null_max, null_label)
            b.axhline(threshold, color='black', linewidth=0.8, linestyle=':')
            b.set_ylabel(r'largest $-\log_{10} p$')
            boxes(d, real_lambda, cross_lambda, null_lambda, null_label)
            d.axhline(1, color='0.3', linewidth=0.8, linestyle=':')
            d.set_ylabel(r'$\lambda_\mathrm{GC}$')
            qq_real_vs_null(c, list(folds[qq_date]['neg_log10_p']),
                            [n['neg_log10_p'][0] for n in nulls],
                            crossfit[qq_date]['neg_log10_p'].ravel())
            for line in c.get_lines():
                if line.get_color() == COLORS['h2opt']:
                    line.set_color(COLOR)
            c.set_ylabel(rf'observed $-\log_{{10}} p$ ({miscanthus.DATES[qq_date]})')
            d.text(1.04, 0.5, population.capitalize(), transform=d.transAxes, rotation=-90,
                   ha='left', va='center', fontsize=ROW_LABEL_SIZE, fontweight='bold')
        for ax, text in zip(axes[0], 'abcd', strict=True):
            letter(ax, text, -0.4 if text == 'c' else -0.25)
        for ax in axes[0, [0, 1, 3]]:
            ax.set_xticklabels([])
        handles = [Line2D([], [], marker='s', color=COLOR, linestyle='', markersize=5),
                   Line2D([], [], marker='D', color=MARK_COLOR, markeredgecolor='white',
                          linestyle='', markersize=4),
                   Line2D([], [], marker='s', color=NULL_LINE_COLOR, linestyle='', markersize=5)]
        fig.legend(handles, ['H2-opt (CNN), 5 cross-validation folds', 'H2-opt (CNN), cross-fit',
                             'permutation null'], loc='upper center', ncol=3, frameon=False,
                   fontsize=TICK_SIZE)
        fig.tight_layout(rect=(0, 0, 0.97, 0.95))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        fig.savefig(out.with_suffix('.png'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    main()
