"""Choose the noise level of sorghum H2-opt by held-out heritability (h2opt.selection).

One level for all genotype folds: h2opt.selection.select_noise_level on all plants, with the
linear model and the training settings of train_h2opt.py. Writes an .npz (levels; scores:
splits x levels; level, the chosen level; noise_sd, its standard deviation) read by
train_h2opt.py, and a figure of the held-out heritability against the level.

    python pipelines/scripts/sorghum/choose_noise.py --out noise.npz --figure noise.pdf \\
        --steps 10000 --device cuda
"""

import argparse

import numpy as np
from h2opt import LinearModel, TraitModels
from h2opt.selection import NOISE_LEVELS, noise_scale, select_noise_level

from h2opt_analysis import sorghum
from h2opt_analysis.figures import plot_noise_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--figure', required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per training run')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    X, genotypes, environment = sorghum.load()
    level, scores = select_noise_level(lambda: TraitModels(1, LinearModel, X.shape[1]), X,
                                       genotypes, environment, n_iter=args.steps,
                                       clip=None, device=args.device)
    noise_sd = level * noise_scale(X)
    np.savez(args.out, levels=NOISE_LEVELS, scores=scores, level=level, noise_sd=noise_sd)
    print('held-out heritability of trait 1 (rows: splits)')
    for a, lev in enumerate(NOISE_LEVELS):
        print(f'  {lev:<6g}', np.round(scores[:, a], 3), f'mean {scores[:, a].mean():.3f}')
    print(f'chosen level {level}, noise SD {noise_sd:.3g}')
    plot_noise_scores(NOISE_LEVELS, scores, level, args.figure)


if __name__ == '__main__':
    main()
