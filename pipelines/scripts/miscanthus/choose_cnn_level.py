"""Choose the noise level of the CNN branch of Miscanthus hybrid H2-opt in each plant fold.

h2opt.selection.score_linear_conv_levels on the training plants of each fold, on one date: linear
H2-opt (the noise of choose_noise.py for the fold and date, linear_steps steps), then the
LinearConvModel with h2opt.ImageConvModel (steps steps) at every level of
h2opt.selection.NOISE_LEVELS, Henderson3 heritability, 5 validation splits of the training plants.
The level of each fold is h2opt.selection.highest_level_within_one_se of its scores, and it serves
all dates. A search on every date costs about 14 x as much (about 20 h of GPU for both
populations), so it runs on one date: the date whose linear trait 1 has the best held-out
heritability in choose_noise.py (mean over the folds and splits at each fold's chosen level), a
choice from the training plants only. Writes an .npz (date; folds; levels; chosen: one level per
fold; scores: folds x splits x levels).

    python pipelines/scripts/miscanthus/choose_cnn_level.py --population central \\
        --images images.npy --noise noise.npz --out cnn_level.npz --linear-steps 10000 \\
        --steps 1000 --max-copies 4 --seed 0 --device cuda
"""

import argparse

import numpy as np
from h2opt import ImageConvModel
from h2opt.selection import NOISE_LEVELS, highest_level_within_one_se, score_linear_conv_levels

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--noise', required=True, help='choose_noise.py output')
    parser.add_argument('--out', required=True)
    parser.add_argument('--linear-steps', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max-copies', type=int,
                        help='models trained at once (GPU memory only; same result)')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    folds = np.arange(miscanthus.N_FOLDS)
    with np.load(args.noise) as f:
        noise_sd, chosen, scores = f['noise_sd'], f['chosen'], f['scores']
    at_chosen = np.array([scores[k][..., NOISE_LEVELS.index(chosen[k])].mean(axis=-1)
                          for k in folds])  # folds x dates
    date = int(np.argmax(at_chosen.mean(axis=0)))
    print(f'date {miscanthus.DATES[date]}: mean held-out heritability of linear trait 1 per date',
          at_chosen.mean(axis=0).round(3), flush=True)
    x = miscanthus.load_date(args.images, design.keep, date)
    subsets = np.array([design.fold != f for f in folds])
    cnn_scores, _ = score_linear_conv_levels(
        lambda: ImageConvModel(x.shape[1]), x, design.families, design.environment,
        noise_sd[:, date], NOISE_LEVELS, subsets=subsets, seed=args.seed, subgroups=design.plots,
        estimator='henderson3', split_units=np.arange(len(x)), n_linear_iter=args.linear_steps,
        n_iter=args.steps, device=args.device, max_copies=args.max_copies)
    levels = np.array([highest_level_within_one_se(NOISE_LEVELS, s) for s in cnn_scores])
    np.savez(args.out, date=date, folds=folds, levels=NOISE_LEVELS, chosen=levels,
             scores=cnn_scores)
    print('chosen CNN level per fold:', levels)
    print('held-out Henderson3 heritability of trait 1, mean over folds and splits:')
    for level, s in zip(NOISE_LEVELS, cnn_scores.mean(axis=(0, 1)), strict=True):
        print(f'  {level:<9.3g} {s:.4f}')


if __name__ == '__main__':
    main()
