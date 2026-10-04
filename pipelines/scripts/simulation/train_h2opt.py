"""Train linear H2-opt on one simulation instance.

The paper's simulation objective: 5 traits, one linear model per trait, Gaussian input noise with
standard deviation 0.02 and no clipping. Trained with h2opt.train's default optimizer (Adam) for
10,000 steps per trait. Writes the model (.pt) and the
traits of all plants, uncorrelated on the training plants (.npz, array 'traits').

    python pipelines/scripts/simulation/train_h2opt.py --spectra spectra.npz \\
        --out-traits h2opt.npz --out-model h2opt.pt --traits 5 --steps 10000 \\
        --seed 0 --device cuda
"""

import argparse

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, synthetic_traits, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--out-model', required=True)
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per trait')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with np.load(args.spectra) as f:
        X, names, test = f['X'], f['names'], f['test']

    model = TraitModels(args.traits, LinearModel, X.shape[1])
    train(model, X, names, None, test.astype(int), model_file=args.out_model,
          n_traits=args.traits, n_iter=args.steps, noise_level=0.02,
          noise='normal', clip=None, device=args.device)
    np.savez(args.out_traits, traits=synthetic_traits(model, X, ~test))


if __name__ == '__main__':
    main()
