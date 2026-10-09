"""Fit a baseline to the Miscanthus images of every date in one fold.

Each baseline is fitted to all pixels of one date on the fold's training plants
(h2opt_analysis.methods.BASELINES). PCH maximizes Henderson3 heritability (families, plots within
families, rep and field column fixed); its ridge is one for all dates, chosen on the training plants
by held-out Henderson3 heritability of the first trait over 5 splits of the plants (PCH.tune over
h2opt.baselines.RIDGES), the mean over the dates. The other baselines are as for sorghum. Writes an
.npz with 'traits' (dates x plants x traits, uncorrelated on the training plants) and, for pch, its
tuning and ridge_traits (ridges x dates x plants x traits), the traits at every ridge.

    python pipelines/scripts/miscanthus/baseline.py --population central --images images.npy \\
        --method pch --fold 0 --traits 1 --out fold0.npz
"""

import argparse

import numpy as np
from h2opt.baselines import PCH, RIDGES
from h2opt.selection import highest_level_within_one_se

from h2opt_analysis import miscanthus
from h2opt_analysis.methods import BASELINES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--method', choices=list(BASELINES), required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    rows = design.fold != args.fold
    families, environment, plots = design.families, design.environment, design.plots
    dates = range(len(miscanthus.DATES))

    def pixels(date):
        x = miscanthus.load_date(args.images, design.keep, date)
        return x.reshape((len(x), -1))

    tuning = {}
    if args.method == 'pch':
        # One date in memory at a time. PCH.tune scores each date on its own and averages the
        # scores over the dates, so tuning date by date and averaging gives the same ridge.
        date_scores, ridge_traits = [], []
        for d in dates:
            x = pixels(d)
            date_scores.append(PCH.tune(
                x[rows], families[rows], environment[rows], subgroups=plots[rows],
                estimator='henderson3', split_units=np.arange(rows.sum()))[2])
            # PCH at every ridge (one decomposition per date), to compare any ridge without
            # fitting again; the traits are those of the chosen ridge
            ridge_traits.append([pch.transform(x) for pch in PCH.fit_ridges(
                x[rows], families[rows], environment[rows], plots[rows], args.traits, RIDGES,
                'henderson3')])
            del x
        fold_scores = np.stack(date_scores).mean(axis=0)
        ridge = highest_level_within_one_se(RIDGES, fold_scores)
        scores = fold_scores.mean(axis=0)
        print(f'ridge {ridge:g}; mean held-out heritability per ridge:',
              ' '.join(f'{s:.3f}' for s in scores), flush=True)
        ridge_traits = np.stack(ridge_traits, axis=1)
        traits = ridge_traits[RIDGES.index(ridge)]
        tuning = dict(ridge=ridge, ridges=RIDGES, ridge_scores=scores,
                      ridge_fold_scores=fold_scores, ridge_traits=ridge_traits)
    else:
        traits = []
        for d in dates:
            x = pixels(d)
            baseline = BASELINES[args.method](args.traits)
            traits.append(baseline.fit(x[rows], families[rows], environment[rows]).transform(x))
    np.savez(args.out, traits=np.stack(traits), **tuning)


if __name__ == '__main__':
    main()
