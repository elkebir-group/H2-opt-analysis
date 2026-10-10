"""Raw features of one dataset with their design, for the lme4 check of the heritability estimator.

Writes a .csv for lme4_heritability.R (the design columns, then one column per feature) and an
.npz with the pipeline estimator's heritability of each feature on all individuals (h2) and the
design columns the lme4 model uses (group, others):
- sorghum: the 2,151 wavelengths; ANOVA among genotypes with the field set as environment.
- miscanthus-central, miscanthus-south: date --date, every third pixel in each direction of the
  6 channels (6 x 21 x 21 = 2,646 features, as the paper); Henderson3 among families, plots
  within families, rep and field column as environment.
- arabidopsis-0d, arabidopsis-6d: the features summed over retention time (MS level x m/z,
  3,000), those with nonzero variance; ANOVA among accessions with the year as environment.

    python pipelines/scripts/shared/estimator_inputs.py --dataset sorghum --out-csv sorghum.csv \\
        --out sorghum.npz
"""

import argparse

import numpy as np
from h2opt import anova_heritability, heritability

from h2opt_analysis import arabidopsis, miscanthus, sorghum

DATASETS = ('sorghum', 'miscanthus-central', 'miscanthus-south', 'arabidopsis-0d',
            'arabidopsis-6d')


def load(args):
    """Features X (n, features), the design as {column: labels} with the group column first, and
    the estimator's heritability of each feature."""
    if args.dataset == 'sorghum':
        X, genotypes, environment = sorghum.load()
        X = np.asarray(X, dtype=np.float64)
        design = {'genotype': genotypes, 'set': environment[:, 0]}
        return X, design, anova_heritability(X, genotypes, environment)
    if args.dataset.startswith('miscanthus'):
        population = args.dataset.split('-')[1]
        d = miscanthus.design(population)
        X = miscanthus.load_date(args.images, d.keep, args.date)[:, :, 1::3, 1::3]
        X = X.reshape((len(X), -1)).astype(np.float64)
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
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    X, design, h2 = load(args)
    names = list(design)
    header = names + [f'f{j}' for j in range(X.shape[1])]
    table = np.column_stack([np.asarray(design[c]).astype(str) for c in names]
                            + [np.char.mod('%.9g', X)])
    np.savetxt(args.out_csv, table, fmt='%s', delimiter=',', header=','.join(header), comments='')
    np.savez(args.out, h2=np.asarray(h2, dtype=np.float64), group=names[0],
             others=','.join(names[1:]))
    print(f'{args.dataset}: {X.shape[0]} individuals, {X.shape[1]} features, design {names}')


if __name__ == '__main__':
    main()
