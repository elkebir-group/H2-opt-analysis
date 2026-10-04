"""Train linear H2-opt on sorghum with the plants of some of the paper's 10 genotype folds held out.

The paper's objective: one linear model per trait, input noise (the paper: uniform up to 0.005),
traits clipped at 2 in the loss, and a penalty on the total variation of the weights across
adjacent wavelengths (the paper: 1e-4; 0 for none). Trained with h2opt.train's default optimizer
(Adam). Writes the model (.pt) and the traits of all plants, uncorrelated on the training plants
(.npz, array 'traits').

    python pipelines/scripts/sorghum/train_h2opt.py --test-folds 0 --out-traits fold0.npz \\
        --out-model fold0.pt --steps 10000 --traits 10 --noise uniform --noise-level 0.005 \\
        --penalty 1e-4 --seed 0 --device cuda
"""

import argparse

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, synthetic_traits, train

from h2opt_analysis import sorghum


def smoothness_penalty(scale):
    """Total variation of the weights over adjacent wavelengths, relative to the trait's spread."""
    def penalty(trait_model, Y):
        w = trait_model.lin1.weight
        variation = torch.mean(torch.sum(torch.abs(w[:, 1:] - w[:, :-1]), axis=1))
        return scale * variation / torch.mean(torch.abs(Y - torch.mean(Y, axis=0)))
    return penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-folds', type=int, nargs='+', required=True,
                        help='genotype folds whose plants are held out')
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--out-model', required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per trait')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--noise', choices=['uniform', 'normal'], required=True)
    parser.add_argument('--noise-level', type=float, required=True,
                        help='maximum of the uniform noise, or standard deviation of the normal')
    parser.add_argument('--penalty', type=float, required=True,
                        help='scale of the smoothness penalty; 0 for none')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X, genotypes, environment = sorghum.load()
    is_train = ~np.isin(sorghum.folds(genotypes), args.test_folds)

    penalty = smoothness_penalty(args.penalty) if args.penalty > 0 else None
    model = TraitModels(args.traits, LinearModel, X.shape[1])
    train(model, X, genotypes, environment, (~is_train).astype(int), model_file=args.out_model,
          n_traits=args.traits, n_iter=args.steps, noise_level=args.noise_level,
          noise=args.noise, penalty=penalty, device=args.device)
    np.savez(args.out_traits, traits=synthetic_traits(model, X, is_train))


if __name__ == '__main__':
    main()
