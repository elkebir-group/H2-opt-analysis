"""NDVI of every kept Miscanthus plant and date, and its Henderson3 heritability per date and fold.

NDVI = (NIR - red) / (NIR + red), with red (channel 3) and NIR (channel 5) the mean of the
pixels of each plant (as the paper: the bands are summed over the pixels first). The images are
the values as received (images.npy: not clipped, not scaled). NDVI is a fixed index, not a trained
method: its heritability is scored as in evaluate_heritability.py (Henderson3 on the training and
the test plants of each fold). Writes an .npz (dates; ndvi: dates x kept plants; train and test:
dates x folds, broad-sense on the plant scale) and prints the mean NDVI and the mean narrow-sense
test heritability (4 x Henderson3, capped at 1) per date.

    python pipelines/scripts/miscanthus/ndvi.py --population central --images images.npy \\
        --out ndvi.npz
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
    train, test = np.zeros((2, n_dates, n_folds))
    for d in range(n_dates):
        for f in range(n_folds):
            for out, use in ((train, design.fold != f), (test, design.fold == f)):
                out[d, f] = heritability(ndvi[d, use, None], design.families[use],
                                         design.environment[use], design.plots[use],
                                         'henderson3')[0]
    np.savez(args.out, dates=miscanthus.DATES, ndvi=ndvi, train=train, test=test)
    print(f'{"date":8s}{"mean NDVI":>12s}{"test h2":>12s}')
    for d, date in enumerate(miscanthus.DATES):
        print(f'{date:8s}{ndvi[d].mean():12.3f}{np.minimum(4 * test[d], 1).mean():12.3f}')


if __name__ == '__main__':
    main()
