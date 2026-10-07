"""Score one noise level of Miscanthus H2-opt (the image CNN) in each fold, on every date.

h2opt.selection.select_noise_level at one level, on the training plants of each fold (all folds in
one run), trait 1 of every date: validation splits of the training plants (split_units: the
plants), the CNN h2opt.ImageConvModel trained with Henderson3 heritability (families, plots within
families, rep and field column fixed) and normal noise, scored by Henderson3 on the held-out
plants. One run per level, so that the grid can be extended without training the other levels
again; choose_noise.py chooses the level from the scores of all levels. Writes an .npz with level;
scores, folds x dates x 1 x splits; scale, folds x dates, h2opt.selection.noise_scale of each
date's images on the training plants (the noise standard deviation is level x scale); and the
trained validation traits, folds x dates x 1 x splits x plants, to examine other rules without
training again.

    python pipelines/scripts/miscanthus/score_noise.py --population central --images images.npy \\
        --level 0.1 --out level0.1.npz --steps 1000 --splits 5 --max-copies 6 --device cuda
"""

import argparse

import numpy as np
from h2opt import ImageConvModel
from h2opt.selection import noise_scale, select_noise_level

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--level', type=float, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per training run')
    parser.add_argument('--splits', type=int, required=True, help='validation splits (of 5)')
    parser.add_argument('--max-copies', type=int, required=True, help='models trained at once')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    images = [miscanthus.load_date(args.images, design.keep, d)
              for d in range(len(miscanthus.DATES))]
    is_train = np.array([design.fold != f for f in range(miscanthus.N_FOLDS)])
    _, scores, traits = select_noise_level(
        lambda: ImageConvModel(images[0].shape[1], images[0].shape[2:]), images, design.families,
        design.environment, 1, subsets=is_train, levels=[args.level], n_splits=args.splits,
        subgroups=design.plots, estimator='henderson3', split_units=np.arange(len(design.fold)),
        n_iter=args.steps, device=args.device, max_copies=args.max_copies)
    scale = np.array([[noise_scale(x[rows].reshape((rows.sum(), -1))) for x in images]
                      for rows in is_train])
    np.savez(args.out, level=args.level, scores=scores[..., 0], scale=scale,
             traits=traits[..., 0, :])
    print(f'level {args.level:g}; held-out heritability per fold (mean over dates and splits):',
          ' '.join(f'{s:.3f}' for s in scores.mean(axis=(1, 2, 3, 4))))


if __name__ == '__main__':
    main()
