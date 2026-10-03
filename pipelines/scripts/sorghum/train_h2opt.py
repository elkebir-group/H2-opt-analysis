"""Train linear H2-opt on sorghum for one of the paper's 10 genotype folds (the paper's settings).

One linear model per trait, RMSprop with learning rate 1e-5, uniform input noise up to 0.005, and a
penalty of 1e-4 on the total variation of the weights across adjacent wavelengths. Writes the model
(.pt) and the traits of all plants, uncorrelated on the training plants (.npz, array 'traits').

    python pipelines/scripts/sorghum/train_h2opt.py --fold 0 --out-traits fold0.npz \\
        --out-model fold0.pt --steps 10000 --traits 10 --seed 0 --device cuda
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
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--out-model', required=True)
    parser.add_argument('--steps', type=int, required=True, help='RMSprop steps per trait')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X, genotypes, environment = sorghum.load()
    is_train = sorghum.folds(genotypes) != args.fold

    model = TraitModels(args.traits, LinearModel, X.shape[1])
    train(model, X, genotypes, environment, (~is_train).astype(int), model_file=args.out_model,
          n_traits=args.traits, n_iter=args.steps, learning_rate=1e-5, noise_level=0.005,
          penalty=smoothness_penalty(1e-4), device=args.device)
    np.savez(args.out_traits, traits=synthetic_traits(model, X, is_train))


if __name__ == '__main__':
    main()
