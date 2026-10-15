"""Pearson correlations of the Miscanthus synthetic traits between dates.

The synthetic traits are trait 1 of one method (one file per date: folds x kept plants x traits),
on all kept plants (training and test plants). Each correlation is computed per
plant and between half-sib family means, in each fold (the model of the fold). Writes an .npz
(dates; plant and family: dates x dates x folds) and prints the median |r| over the folds of the
family means of the first five dates.

    python pipelines/scripts/miscanthus/date_correlations.py --population central \\
        --traits date0.npz ... date13.npz --out date_correlations.npz
"""

import argparse

import numpy as np

from h2opt_analysis import miscanthus


def family_means(values, families):
    """Mean of values over the plants of each family (sorted family order)."""
    _, inverse = np.unique(families, return_inverse=True)
    return np.bincount(inverse, values) / np.bincount(inverse)


def pearson(a, b):
    return np.corrcoef(a, b)[0, 1]


def load_traits(files):
    """Trait 1 of one file per date: dates x folds x kept plants."""
    assert len(files) == len(miscanthus.DATES), 'give one file per date'
    return np.stack([np.load(f)['traits'][..., 0] for f in files])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--traits', nargs='+', required=True, help='one file per date')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    traits = load_traits(args.traits)
    n_dates, n_folds = traits.shape[:2]
    means = np.stack([[family_means(traits[d, f], design.families) for f in range(n_folds)]
                      for d in range(n_dates)])
    plant, family = np.zeros((2, n_dates, n_dates, n_folds))
    for d1 in range(n_dates):
        for d2 in range(n_dates):
            for f in range(n_folds):
                plant[d1, d2, f] = pearson(traits[d1, f], traits[d2, f])
                family[d1, d2, f] = pearson(means[d1, f], means[d2, f])
    np.savez(args.out, dates=miscanthus.DATES, plant=plant, family=family)
    print('median |r| over folds of family means between dates')
    print(np.round(np.median(np.abs(family), axis=2)[:5, :5], 2))


if __name__ == '__main__':
    main()
