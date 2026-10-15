"""Raw features of one dataset with their design, for the lme4 check of the heritability estimator.

Writes a .csv for lme4_heritability.R (the design columns, then one column per feature) and an
.npz with the pipeline estimator's heritability of each feature on all individuals (h2):
- sorghum: the 2,151 wavelengths; ANOVA among genotypes with the field set as environment.
- miscanthus-central, miscanthus-south: date --date, every third pixel in each direction of the
  6 channels (6 x 21 x 21 = 2,646 features), after the nearest-neighbor field
  adjustment (--width, slope fitted on all plants); Henderson3 among families, plots within
  families, rep and field column as environment.
- arabidopsis-0d, arabidopsis-6d: the features summed over retention time (MS level x m/z,
  3,000), those with nonzero variance; ANOVA among accessions with the year as environment.
- miscanthus-traits-central, miscanthus-traits-south: trait 1 of hybrid H2-opt trained after the
  nearest-neighbor adjustment (--traits, one file per date) of every date and fold, on the fold's
  test plants only (missing elsewhere), after the nearest-neighbor adjustment with the slope fitted
  on the test plants; Henderson3 as for the pixels.

    python pipelines/scripts/shared/estimator_inputs.py --dataset sorghum --out-csv sorghum.csv \\
        --out sorghum.npz
"""

import argparse

import numpy as np
from h2opt import anova_heritability, heritability

from h2opt_analysis import arabidopsis, miscanthus, sorghum

DATASETS = ('sorghum', 'miscanthus-central', 'miscanthus-south', 'arabidopsis-0d',
            'arabidopsis-6d', 'miscanthus-traits-central', 'miscanthus-traits-south')


def load(args):
    """Features X (n, features), the design as {column: labels} with the group column first, and
    the estimator's heritability of each feature."""
    if args.dataset == 'sorghum':
        X, genotypes, environment = sorghum.load()
        X = np.asarray(X, dtype=np.float64)
        design = {'genotype': genotypes, 'set': environment[:, 0]}
        return X, design, anova_heritability(X, genotypes, environment)
    if args.dataset.startswith('miscanthus-traits'):
        population = args.dataset.split('-')[2]
        d = miscanthus.design(population)
        weights = miscanthus.neighbor_weights(d, args.width)
        traits = np.stack([np.load(f)['traits'][:, :, 0] for f in args.traits])  # dates x folds
        X = np.full((len(d.families), traits.shape[0] * traits.shape[1]), np.nan)
        h2 = np.zeros(X.shape[1])
        for j, (date, fold) in enumerate(np.ndindex(traits.shape[:2])):
            test = d.fold == fold
            X[test, j] = miscanthus.papadakis(traits[date, fold], weights, test)[test]
            h2[j] = heritability(X[test, j], d.families[test], d.environment[test], d.plots[test],
                                 'henderson3')[0]
        design = {'family': d.families, 'plot': d.plots, 'rep': d.environment[:, 0],
                  'column': d.environment[:, 1]}
        return X, design, h2
    if args.dataset.startswith('miscanthus'):
        population = args.dataset.split('-')[1]
        d = miscanthus.design(population)
        X = miscanthus.load_date(args.images, d.keep, args.date)[:, :, 1::3, 1::3]
        X = X.reshape((len(X), -1)).astype(np.float64)
        X = miscanthus.papadakis(X, miscanthus.neighbor_weights(d, args.width),
                                 np.ones(len(X), dtype=bool))
        design = {'family': d.families, 'plot': d.plots, 'rep': d.environment[:, 0],
                  'column': d.environment[:, 1]}
        return X, design, heritability(X, d.families, d.environment, d.plots, 'henderson3')
    treatment = args.dataset.split('-')[1]
    X, accessions, environment = arabidopsis.load(args.features, treatment)
    X = X.reshape((len(X), 2 * arabidopsis.N_MZ, arabidopsis.N_RT)).sum(axis=2)
    X = X[:, X.std(axis=0) > 0].astype(np.float64)
    design = {'accession': accessions, 'year': environment[:, 0]}
    return X, design, anova_heritability(X, accessions, environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--images', help='miscanthus: images.npy of the population')
    parser.add_argument('--date', type=int, help='miscanthus: index into miscanthus.DATES')
    parser.add_argument('--features', help='arabidopsis: features.npy')
    parser.add_argument('--width', type=int,
                        help='miscanthus: field columns of the nearest-neighbor adjustment')
    parser.add_argument('--traits', nargs='+', help='miscanthus-traits: one traits file per date')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    X, design, h2 = load(args)
    names = list(design)
    header = names + [f'f{j}' for j in range(X.shape[1])]
    table = np.column_stack([np.asarray(design[c]).astype(str) for c in names]
                            + [np.where(np.isnan(X), 'NA', np.char.mod('%.9g', X))])
    np.savetxt(args.out_csv, table, fmt='%s', delimiter=',', header=','.join(header), comments='')
    np.savez(args.out, h2=np.asarray(h2, dtype=np.float64))
    print(f'{args.dataset}: {X.shape[0]} individuals, {X.shape[1]} features, design {names}')


if __name__ == '__main__':
    main()
