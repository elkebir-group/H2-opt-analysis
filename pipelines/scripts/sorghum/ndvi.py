"""NDVI of every sorghum plant and its train and test heritability per genotype fold.

NDVI = (NIR - red) / (NIR + red) (Sims and Gamon, 2002), with red the mean reflectance over the
20 wavelengths 670-689 nm and NIR the mean over 790-809 nm (sorghum.ndvi; the spectra are sampled
at 1 nm from 350 nm). NDVI is a fixed index, not a trained method: its heritability is scored as in
evaluate_heritability.py (anova_heritability with the field set, on the training and the test
plants of each fold). Writes an .npz (folds; ndvi: plants; train and test: folds) and prints the
heritability per fold and the means.

    python pipelines/scripts/sorghum/ndvi.py --folds 0 1 2 3 4 5 6 7 8 9 --out ndvi.npz
"""

import argparse

import numpy as np
from h2opt import anova_heritability

from h2opt_analysis import sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    X, genotypes, environment = sorghum.load()
    ndvi = sorghum.ndvi(X)
    assert np.isfinite(ndvi).all(), 'NDVI is not finite for some plants'
    fold_of_plant = sorghum.folds(genotypes)
    train, test = np.zeros((2, len(args.folds)))
    for k, fold in enumerate(args.folds):
        for out, rows in ((train, fold_of_plant != fold), (test, fold_of_plant == fold)):
            out[k] = anova_heritability(ndvi[rows, None], genotypes[rows], environment[rows])[0]
        print(f'fold {fold}: train {train[k]:.3f}, test {test[k]:.3f}', flush=True)

    np.savez(args.out, folds=args.folds, ndvi=ndvi, train=train, test=test)
    print(f'\nmean NDVI {ndvi.mean():.3f}')
    print(f'mean train heritability {train.mean():.3f}')
    print(f'mean test heritability {test.mean():.3f} (range {test.min():.3f}-{test.max():.3f})')


if __name__ == '__main__':
    main()
