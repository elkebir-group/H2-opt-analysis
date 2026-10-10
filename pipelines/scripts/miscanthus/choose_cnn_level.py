"""Choose the one CNN noise level of Miscanthus hybrid H2-opt: all folds and all dates.

h2opt.selection.score_linear_conv_levels on the training plants of one fold (SEARCH_FOLD, fixed in
advance; one fold instead of all 5 keeps the search at about 80 min), on the date of
choose_noise.py: linear H2-opt (the one linear level of choose_noise.py, linear_steps steps), then
the LinearConvModel with h2opt.PositionImageConvModel (steps steps; a weight per position: the plant
is at the center, its neighbors at the edges) at every level of h2opt.selection.NOISE_LEVELS,
Henderson3 heritability, 20 validation splits of the training plants (4 partitions into 5 folds).
The CNN level is h2opt.selection.highest_level_within_one_se of the scores; it is separate from the
linear level (the two levels act on different branches and are not comparable). Writes an .npz
(date; fold; levels; chosen: the level; scores: splits x levels).

    python pipelines/scripts/miscanthus/choose_cnn_level.py --population central \\
        --images images.npy --noise noise.npz --out cnn_level.npz --linear-steps 10000 \\
        --steps 1000 --max-copies 4 --seed 0 --device cuda
"""

import argparse

import numpy as np
from h2opt import PositionImageConvModel
from h2opt.selection import NOISE_LEVELS, highest_level_within_one_se, score_linear_conv_levels

from h2opt_analysis import miscanthus

SEARCH_FOLD = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--noise', required=True, help='choose_noise.py output')
    parser.add_argument('--out', required=True)
    parser.add_argument('--linear-steps', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max-copies', type=int,
                        help='models trained at once (GPU memory)')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    with np.load(args.noise) as f:
        noise_sd, date = f['noise_sd'][SEARCH_FOLD], int(f['date'])
    x = miscanthus.load_date(args.images, design.keep, date)
    scores, _ = score_linear_conv_levels(
        lambda: PositionImageConvModel(x.shape[1], x.shape[2:]), x, design.families,
        design.environment, noise_sd[date], NOISE_LEVELS,
        subsets=(design.fold != SEARCH_FOLD)[None], seed=args.seed, subgroups=design.plots,
        estimator='henderson3',
        split_units=np.arange(len(x)), n_linear_iter=args.linear_steps, n_iter=args.steps,
        device=args.device, max_copies=args.max_copies)
    chosen = highest_level_within_one_se(NOISE_LEVELS, scores[0])
    np.savez(args.out, date=date, fold=SEARCH_FOLD, levels=NOISE_LEVELS, chosen=chosen,
             scores=scores[0])
    print(f'date {miscanthus.DATES[date]}, fold {SEARCH_FOLD}; chosen CNN level {chosen:g}')
    print('held-out Henderson3 heritability of trait 1, mean over splits:')
    for level, v in zip(NOISE_LEVELS, scores[0].mean(axis=0), strict=True):
        print(f'  {level:<9.3g} {v:.4f}')


if __name__ == '__main__':
    main()
