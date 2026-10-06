"""Choose the noise level of Miscanthus H2-opt (the image CNN) in each fold, one for all dates.

h2opt.selection.select_noise_level on the training plants of each fold (all folds in one
selection), trait 1 of every date: validation splits of the training plants (split_units: the
plants), the CNN h2opt.ImageConvModel trained with Henderson3 heritability (families, plots
within families, rep and field column fixed) and normal noise, scored by Henderson3 on the
held-out plants; the level with the best score averaged over the dates and splits. Writes an .npz
(levels; chosen: one level per fold; noise_sd: folds x dates, the standard deviation of the
chosen noise on each date's images; scores: folds x dates x 1 x splits x levels) read by
train_h2opt.py; an .npz of the trained validation traits (traits: folds x dates x 1 x splits x
levels x plants), to examine other rules without training again; and a figure of the choices.

    python pipelines/scripts/miscanthus/choose_noise.py --population central --images images.npy \\
        --out noise.npz --out-traits noise_traits.npz --figure noise.pdf --steps 1000 \\
        --levels 0.01 0.1 1 --splits 5 --max-copies 6 --device cuda
"""

import argparse

import numpy as np
from h2opt import ImageConvModel
from h2opt.selection import noise_scale, select_noise_level

from h2opt_analysis import miscanthus
from h2opt_analysis.figures import plot_noise_choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--figure', required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per training run')
    parser.add_argument('--levels', type=float, nargs='+', required=True)
    parser.add_argument('--splits', type=int, required=True, help='validation splits (of 5)')
    parser.add_argument('--max-copies', type=int, required=True, help='models trained at once')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    images = [miscanthus.load_date(args.images, design.keep, d)
              for d in range(len(miscanthus.DATES))]
    folds = np.arange(miscanthus.N_FOLDS)
    is_train = np.array([design.fold != f for f in folds])
    chosen, scores, traits = select_noise_level(
        lambda: ImageConvModel(images[0].shape[1], images[0].shape[2:]), images, design.families,
        design.environment, 1, subsets=is_train, levels=args.levels, n_splits=args.splits,
        subgroups=design.plots, estimator='henderson3', split_units=np.arange(len(design.fold)),
        n_iter=args.steps, device=args.device, max_copies=args.max_copies)
    noise_sd = np.array([[level * noise_scale(x[rows].reshape((rows.sum(), -1))) for x in images]
                         for level, rows in zip(chosen, is_train, strict=True)])
    np.savez(args.out, folds=folds, dates=miscanthus.DATES, levels=args.levels, chosen=chosen,
             noise_sd=noise_sd, scores=scores)
    np.savez(args.out_traits, folds=folds, levels=args.levels, traits=traits)
    print('chosen noise level; held-out heritability per level (mean over dates and splits)')
    print('  levels:', ' '.join(f'{lev:.3g}' for lev in args.levels))
    for fold, level, score in zip(folds, chosen, scores, strict=True):
        means = score.mean(axis=(0, 1, 2))
        print(f'  fold {fold}: {level:.3g};', ' '.join(f'{m:.3f}' for m in means))
    plot_noise_choices(args.levels, chosen, scores[:, :, 0], args.figure, series='date',
                       series_labels=miscanthus.DATES)


if __name__ == '__main__':
    main()
