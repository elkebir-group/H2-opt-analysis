"""Figure of the lme4 check: the pipeline estimator's heritability of each raw feature against
lme4's (lme4_heritability.R), one lettered panel per dataset with the Pearson r, in the rows
of ROWS (paper Fig. ANOVA), drawn at the text width of the paper (6.5 in). The H2-opt trait
panels clip the estimator at 0, as in miscFig; lme4 cannot go below
0 either.

    python pipelines/scripts/shared/plot_estimator_check.py \\
        --dataset sorghum sorghum.npz sorghum_lme4.csv --dataset ... --out ANOVA.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis.figures import MATH_RC, SIM_FIG_RC, letter  # noqa: E402

LABELS = {'sorghum': 'Sorghum (ANOVA)',
          'miscanthus-central': "Miscanthus Central\npixels\n(Henderson's Method III)",
          'miscanthus-south': "Miscanthus South\npixels\n(Henderson's Method III)",
          'arabidopsis-0d': 'Arabidopsis 0 day (ANOVA)',
          'arabidopsis-6d': 'Arabidopsis 6 day (ANOVA)',
          'miscanthus-traits-central': ('Miscanthus Central\n'
                                        "H2-opt traits\n(Henderson's Method III,\nclipped at 0)"),
          'miscanthus-traits-south': ('Miscanthus South\n'
                                      "H2-opt traits\n(Henderson's Method III,\nclipped at 0)")}
CLIPPED = {'miscanthus-traits-central', 'miscanthus-traits-south'}
ROWS = [['sorghum', 'arabidopsis-0d', 'arabidopsis-6d'],
        ['miscanthus-central', 'miscanthus-south', 'miscanthus-traits-central',
         'miscanthus-traits-south']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs=3, action='append', required=True,
                        metavar=('NAME', 'ESTIMATOR_NPZ', 'LME4_CSV'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    files = {name: (estimator, lme4) for name, estimator, lme4 in args.dataset}
    width = max(len(row) for row in ROWS)
    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig, axes = plt.subplots(len(ROWS), width, figsize=(6.5, 3.9), squeeze=False)
        letters = iter('abcdefghijklmnopqrstuvwxyz')
        for row, names in zip(axes, ROWS, strict=True):
            for ax in row[len(names):]:
                ax.set_visible(False)
            row[0].set_ylabel('estimator heritability')
            for ax, name in zip(row[:len(names)], names, strict=True):
                estimator, lme4 = files[name]
                panel(ax, name, estimator, lme4, next(letters))
        fig.tight_layout(h_pad=1.2)
        fig.savefig(args.out)
        fig.savefig(args.out.removesuffix('.pdf') + '.png', dpi=300)


def panel(ax, name, estimator, lme4, text):
    """One dataset: estimator against lme4 heritability, the diagonal and the Pearson r."""
    ours = np.load(estimator)['h2']
    if name in CLIPPED:
        ours = np.clip(ours, 0, None)
    theirs = np.genfromtxt(lme4, delimiter=',', skip_header=1, usecols=1)
    ok = np.isfinite(ours) & np.isfinite(theirs)
    r = np.corrcoef(theirs[ok], ours[ok])[0, 1]
    ax.scatter(theirs[ok], ours[ok], s=1.5, alpha=0.5, linewidth=0, rasterized=True)
    top = max(theirs[ok].max(), ours[ok].max())
    ax.plot([0, top], [0, top], 'k:', linewidth=0.8)
    ax.set_title(f'{LABELS[name]}\n$r = {r:.3f}$', fontsize=6)
    letter(ax, text, -0.35)
    ax.set_xlabel('lme4 heritability')
    ax.spines[['top', 'right']].set_visible(False)
    print(f'{name}: {ok.sum()} of {len(ok)} features, Pearson r {r:.4f}, '
          f'mean difference {np.mean(ours[ok] - theirs[ok]):+.4f}')


if __name__ == '__main__':
    main()
