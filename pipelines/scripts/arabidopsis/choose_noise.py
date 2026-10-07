"""Choose the noise level of Arabidopsis H2-opt in each accession fold, one for all traits.

h2opt.selection.select_noise_level on the training samples of each fold of one treatment (all
folds in one batch), with the linear model and the training settings of train_h2opt.py: 5
validation splits of the training accessions; the level whose traits have the best held-out
heritability averaged over the traits and splits. The test samples of a fold never take part in
its choice. Writes an .npz (folds; levels; chosen: one level per fold; noise_sd: the standard
deviation of the chosen noise per fold; scores: folds x traits x splits x levels) read by
train_h2opt.py; an .npz of the trained validation traits (traits: folds x traits x splits x
levels x samples, before decorrelation); and a figure of the choices.

    python pipelines/scripts/arabidopsis/choose_noise.py --features features.npy --treatment 6d \\
        --folds 0 1 2 3 4 --out noise.npz --out-traits noise_traits.npz --figure noise.pdf \\
        --traits 10 --steps 16000 --device cuda
"""

import argparse

import numpy as np
from h2opt import LinearModel
from h2opt.selection import NOISE_LEVELS, best_level, select_noise_level

from h2opt_analysis import arabidopsis
from h2opt_analysis.figures import plot_noise_choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
    parser.add_argument('--folds', type=int, nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--figure', required=True)
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per training run')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    X, accessions, environment = arabidopsis.load(args.features, args.treatment)
    fold_of = arabidopsis.folds(accessions)
    is_train = np.array([fold_of != fold for fold in args.folds])
    noise_sd, scores, traits = select_noise_level(lambda: LinearModel(X.shape[1]), X, accessions,
                                                  environment, args.traits, subsets=is_train,
                                                  n_iter=args.steps, device=args.device)
    chosen = np.array([best_level(NOISE_LEVELS, score) for score in scores])
    np.savez(args.out, folds=args.folds, levels=NOISE_LEVELS, chosen=chosen, noise_sd=noise_sd,
             scores=scores)
    np.savez(args.out_traits, folds=args.folds, levels=NOISE_LEVELS, traits=traits)
    print('chosen noise level; mean held-out heritability per level (over traits and splits)')
    print('  levels:', ' '.join(f'{lev:.3g}' for lev in NOISE_LEVELS))
    for fold, level, score in zip(args.folds, chosen, scores, strict=True):
        print(f'  fold {fold}: {level:.3g};', ' '.join(f'{m:.3f}' for m in score.mean(axis=(0, 1))))
    plot_noise_choices(NOISE_LEVELS, chosen, scores, args.figure)


if __name__ == '__main__':
    main()
