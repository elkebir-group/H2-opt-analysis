"""Sensitivity of H2-opt to the noise level on one simulation instance.

H2-opt as in train_h2opt.py, trained once for each level of h2opt.selection.NOISE_LEVELS (normal
noise with standard deviation level * noise_scale of the training spectra), all levels at once by
h2opt.train_models from the same initial model. Writes an .npz (levels; noise_sd per level;
traits: levels x plants x traits, each uncorrelated on the training plants).

    python pipelines/scripts/simulation/noise_sweep.py --spectra spectra.npz --out sweep.npz \\
        --traits 5 --steps 10000 --seed 0 --device cuda
"""

import argparse
import copy

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, synthetic_traits, train_models
from h2opt.selection import NOISE_LEVELS, noise_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per trait')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with np.load(args.spectra) as f:
        X, names, test = f['X'], f['names'], f['test']
    noise_sd = np.array(NOISE_LEVELS) * noise_scale(X[~test])

    start = TraitModels(args.traits, LinearModel, X.shape[1])
    models = [copy.deepcopy(start) for _ in NOISE_LEVELS]
    train_models(models, X, names, None, np.tile(test.astype(int), (len(models), 1)),
                 n_traits=args.traits, n_iter=args.steps, noise_sd=noise_sd[:, None],
                 device=args.device)
    traits = np.array([synthetic_traits(model, X, ~test) for model in models])
    np.savez(args.out, levels=NOISE_LEVELS, noise_sd=noise_sd, traits=traits)


if __name__ == '__main__':
    main()
