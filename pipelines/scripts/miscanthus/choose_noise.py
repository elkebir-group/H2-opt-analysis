"""Choose the noise level of Miscanthus linear H2-opt in each plant fold, one for all dates.

h2opt.selection.select_noise_level on the training plants of each fold (all folds in one batch):
h2opt.LinearModel on all pixels of one date (h2opt_analysis.miscanthus.load_date, flattened),
Henderson3 heritability (families, plots within families, rep and field column fixed), 5
validation splits of the training plants, every level of h2opt.selection.NOISE_LEVELS, the first
trait trained for steps steps. Every date is trained and scored; one level per fold serves all
dates, by the mean score over the dates (h2opt.selection.highest_level_within_one_se). The test
plants of a fold never take part in its choice. Writes an .npz (folds; dates; levels; chosen: one
level per fold; noise_sd: folds x dates, the chosen level times noise_scale of the fold's training
pixels of each date; scores: folds x dates x splits x levels).

    python pipelines/scripts/miscanthus/choose_noise.py --population central --images images.npy \\
        --out noise.npz --steps 10000 --max-copies 105 --device cuda
"""

import argparse

import numpy as np
from h2opt import LinearModel
from h2opt.selection import NOISE_LEVELS, highest_level_within_one_se, select_noise_level

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max-copies', type=int,
                        help='models trained at once (GPU memory only; same result)')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    folds = np.arange(miscanthus.N_FOLDS)
    subsets = np.array([design.fold != f for f in folds])
    X = []
    for d in range(len(miscanthus.DATES)):
        x = miscanthus.load_date(args.images, design.keep, d)
        X.append(x.reshape((len(x), -1)))
    noise_sd, scores, _ = select_noise_level(
        lambda: LinearModel(X[0].shape[1]), X, design.families, design.environment, subsets,
        subgroups=design.plots, estimator='henderson3', split_units=np.arange(len(X[0])),
        n_iter=args.steps, device=args.device, max_copies=args.max_copies)
    chosen = np.array([highest_level_within_one_se(NOISE_LEVELS, s) for s in scores])
    np.savez(args.out, folds=folds, dates=miscanthus.DATES, levels=NOISE_LEVELS, chosen=chosen,
             noise_sd=noise_sd, scores=scores)
    print('chosen level per fold:', chosen)
    print('held-out Henderson3 heritability of trait 1, mean over folds, dates and splits:')
    for level, s in zip(NOISE_LEVELS, scores.mean(axis=(0, 1, 2)), strict=True):
        print(f'  {level:<9.3g} {s:.4f}')


if __name__ == '__main__':
    main()
