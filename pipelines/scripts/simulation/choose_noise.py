"""Choose the noise level of H2-opt for one simulation instance (h2opt.selection).

h2opt.selection.select_noise_level on the training plants of the instance, with the linear model
and the training settings of train_h2opt.py. Writes an .npz (levels; scores: splits x levels;
level, the chosen level; noise_sd, its standard deviation) read by train_h2opt.py.

    python pipelines/scripts/simulation/choose_noise.py --spectra spectra.npz --out noise.npz \\
        --steps 10000 --seed 0 --device cuda
"""

import argparse

import numpy as np
from h2opt import LinearModel, TraitModels
from h2opt.selection import NOISE_LEVELS, noise_scale, select_noise_level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per training run')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    with np.load(args.spectra) as f:
        is_train = ~f['test']
        X, names = f['X'][is_train], f['names'][is_train]
    level, scores = select_noise_level(lambda: TraitModels(1, LinearModel, X.shape[1]), X,
                                       names, None, seed=args.seed, n_iter=args.steps,
                                       clip=None, device=args.device)
    noise_sd = level * noise_scale(X)
    np.savez(args.out, levels=NOISE_LEVELS, scores=scores, level=level, noise_sd=noise_sd)
    print('held-out heritability of trait 1 (rows: splits)')
    for a, lev in enumerate(NOISE_LEVELS):
        print(f'  {lev:<6g}', np.round(scores[:, a], 3), f'mean {scores[:, a].mean():.3f}')
    print(f'chosen level {level}, noise SD {noise_sd:.3g}')


if __name__ == '__main__':
    main()
