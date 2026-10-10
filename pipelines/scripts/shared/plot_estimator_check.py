"""Figure of the lme4 check: the pipeline estimator's heritability of each raw feature against
lme4's (lme4_heritability.R), one panel per dataset, with the Pearson r (paper Fig. ANOVA).

    python pipelines/scripts/shared/plot_estimator_check.py \\
        --dataset sorghum sorghum.npz sorghum_lme4.csv --dataset ... --out ANOVA.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {'sorghum': 'sorghum (ANOVA)', 'miscanthus-central': 'Miscanthus Central (Henderson3)',
          'miscanthus-south': 'Miscanthus South (Henderson3)',
          'arabidopsis-0d': 'Arabidopsis 0 day (ANOVA)',
          'arabidopsis-6d': 'Arabidopsis 6 day (ANOVA)',
          'miscanthus-traits-central': 'Miscanthus Central H2-opt traits (Henderson3)',
          'miscanthus-traits-south': 'Miscanthus South H2-opt traits (Henderson3)'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs=3, action='append', required=True,
                        metavar=('NAME', 'ESTIMATOR_NPZ', 'LME4_CSV'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, len(args.dataset), figsize=(3 * len(args.dataset), 3.4))
    for ax, (name, estimator, lme4) in zip(np.atleast_1d(axes), args.dataset, strict=True):
        ours = np.load(estimator)['h2']
        theirs = np.genfromtxt(lme4, delimiter=',', skip_header=1, usecols=1)
        ok = np.isfinite(ours) & np.isfinite(theirs)
        r = np.corrcoef(theirs[ok], ours[ok])[0, 1]
        ax.scatter(theirs[ok], ours[ok], s=3, alpha=0.5, rasterized=True)
        top = max(theirs[ok].max(), ours[ok].max())
        ax.plot([0, top], [0, top], 'k:', linewidth=1)
        ax.set_title(f'{LABELS[name]}\nr = {r:.3f}', fontsize=8)
        ax.set_xlabel('lme4 heritability')
        ax.spines[['top', 'right']].set_visible(False)
        print(f'{name}: {ok.sum()} of {len(ok)} features, Pearson r {r:.4f}, '
              f'mean difference {np.mean(ours[ok] - theirs[ok]):+.4f}')
    np.atleast_1d(axes)[0].set_ylabel('pipeline estimator heritability')
    fig.tight_layout()
    fig.savefig(args.out)
    fig.savefig(args.out.removesuffix('.pdf') + '.png', dpi=150)


if __name__ == '__main__':
    main()
