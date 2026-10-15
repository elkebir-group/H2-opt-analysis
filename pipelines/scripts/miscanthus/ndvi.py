"""NDVI of every kept Miscanthus plant and date, and its Henderson3 heritability per date and fold.

NDVI = (NIR - red) / (NIR + red), with red (channel 3) and NIR (channel 5) the mean of the
pixels of each plant (the bands are summed over the pixels first). The images are
the values as received (images.npy: not clipped, not scaled). NDVI is a fixed index, not a trained
method: its heritability is scored by Henderson3 on the training and the test plants of each fold,
and on the test plants after the nearest-neighbor field adjustment of the other methods
(evaluate_heritability_spatial.py; --width field columns, the slope fitted on
the test plants). Writes an .npz (dates; ndvi: dates x kept plants; train, test and
test_adjusted: dates x folds, Henderson3) and prints the mean NDVI and the mean narrow-sense test
heritability (4 x Henderson3, capped at 1) per date, before and after the adjustment.

    python pipelines/scripts/miscanthus/ndvi.py --population central --images images.npy \\
        --width 3 --out ndvi.npz
"""

import argparse

import numpy as np
from h2opt import heritability

from h2opt_analysis import miscanthus

RED, NIR = 3, 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--width', type=int, required=True, help='neighbor columns on each side')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    images = np.load(args.images, mmap_mode='r')
    rows = np.flatnonzero(design.keep)
    n_dates, n_folds = len(miscanthus.DATES), miscanthus.N_FOLDS
    ndvi = np.zeros((n_dates, len(rows)))
    for d in range(n_dates):
        bands = np.asarray(images[rows, d][:, [RED, NIR]], dtype=np.float64).mean(axis=(2, 3))
        ndvi[d] = (bands[:, 1] - bands[:, 0]) / (bands[:, 1] + bands[:, 0])
    assert np.isfinite(ndvi).all(), 'NDVI is not finite for some plants'
    weights = miscanthus.neighbor_weights(design, args.width)

    def h3(y, use):
        return float(np.ravel(heritability(y[use, None], design.families[use],
                                           design.environment[use], design.plots[use],
                                           'henderson3'))[0])

    train, test, test_adjusted = np.zeros((3, n_dates, n_folds))
    for d in range(n_dates):
        for f in range(n_folds):
            use = design.fold == f
            train[d, f] = h3(ndvi[d], design.fold != f)
            test[d, f] = h3(ndvi[d], use)
            test_adjusted[d, f] = h3(miscanthus.papadakis(ndvi[d], weights, use), use)
    np.savez(args.out, dates=miscanthus.DATES, ndvi=ndvi, train=train, test=test,
             test_adjusted=test_adjusted)
    print(f'{"date":8s}{"mean NDVI":>12s}{"test h2":>12s}{"adjusted":>12s}')
    for d, date in enumerate(miscanthus.DATES):
        print(f'{date:8s}{ndvi[d].mean():12.3f}{np.minimum(4 * test[d], 1).mean():12.3f}'
              f'{np.clip(4 * test_adjusted[d], 0, 1).mean():12.3f}')


if __name__ == '__main__':
    main()
