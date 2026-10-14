"""Choose the one linear noise level of Miscanthus H2-opt: all folds, all dates, linear H2-opt and
the linear branch of the hybrid.

h2opt.selection.select_noise_level on the training plants of each fold (all folds in one batch),
on one date (--date): h2opt.LinearModel on all pixels of the date (h2opt_analysis.miscanthus.
load_date, flattened), Henderson3 heritability (families, plots within families, rep and field
column fixed), 20 validation splits of the training plants (4 partitions into 5 folds), every
level of
h2opt.selection.NOISE_LEVELS, the first trait trained for steps steps. The rule
h2opt.selection.highest_level_within_one_se takes the 100 validation splits of all folds together
and gives one level, the same for all folds (a level per fold follows the noise of the flat,
jagged curves of Central). The level serves all dates, as the level times noise_scale of the
fold's training pixels of each date (the CNN branch has its own level, choose_cnn_level.py). A
validation split of one fold uses training plants that are test plants of the other folds, so the
test plants take a small part in the choice of the one level. Writes an .npz (date; folds; dates;
levels; chosen: the level; noise_sd: folds x dates; scores: folds x splits x levels).

    python pipelines/scripts/miscanthus/choose_noise.py --population central --images images.npy \\
        --date 2 --out noise.npz --steps 10000 --max-copies 105 --device cuda
"""

import argparse

import numpy as np
from h2opt import LinearModel
from h2opt.selection import (
    NOISE_LEVELS,
    highest_level_within_one_se,
    noise_scale,
    select_noise_level,
)

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--date', type=int, required=True,
                        help='index into miscanthus.DATES of the date of the search')
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max-copies', type=int,
                        help='models trained at once (GPU memory)')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    parser.add_argument('--neighbor-width', type=int,
                        help='train on the heritability after the nearest-neighbor field '
                             'adjustment (h2opt.spatial), neighbors within this many columns')
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    folds = np.arange(miscanthus.N_FOLDS)
    subsets = np.array([design.fold != f for f in folds])
    neighbors = (None if args.neighbor_width is None
                 else miscanthus.neighbor_weights(design, args.neighbor_width))

    def pixels(date):
        x = miscanthus.load_date(args.images, design.keep, date)
        return x.reshape((len(x), -1))

    x = pixels(args.date)
    _, scores, _ = select_noise_level(
        lambda: LinearModel(x.shape[1]), x, design.families, design.environment, subsets,
        subgroups=design.plots, estimator='henderson3', split_units=np.arange(len(x)),
        n_iter=args.steps, device=args.device, max_copies=args.max_copies,
        neighbors=neighbors)
    chosen = highest_level_within_one_se(NOISE_LEVELS, scores.reshape((-1, len(NOISE_LEVELS))))
    noise_sd = np.zeros((len(folds), len(miscanthus.DATES)))
    for d in range(len(miscanthus.DATES)):
        x = pixels(d)
        noise_sd[:, d] = [chosen * noise_scale(x[subsets[f]]) for f in folds]
    np.savez(args.out, date=args.date, folds=folds, dates=miscanthus.DATES, levels=NOISE_LEVELS,
             chosen=chosen, noise_sd=noise_sd, scores=scores)
    print(f'date {miscanthus.DATES[args.date]}; chosen level {chosen:g}')
    print('held-out Henderson3 heritability of trait 1, mean over folds and splits:')
    for level, s in zip(NOISE_LEVELS, scores.mean(axis=(0, 1)), strict=True):
        print(f'  {level:<9.3g} {s:.4f}')


if __name__ == '__main__':
    main()
