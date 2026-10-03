"""Train linear H2-opt on sorghum for one of the paper's 10 genotype folds (the paper's settings).

One linear model per trait, RMSprop with learning rate 1e-5, uniform input noise up to 0.005, and a
penalty of 1e-4 on the total variation of the weights across adjacent wavelengths. Writes the model
(.pt) and the traits of all plants (.npz, array 'traits').

    python pipelines/scripts/sorghum/train_h2opt.py --fold 0 --out-traits fold0.npz \\
        --out-model fold0.pt [--steps 10000] [--traits 10] [--seed 0] [--device cuda]
"""

import argparse

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, train

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
    parser.add_argument('--steps', type=int, default=10000, help='RMSprop steps per trait')
    parser.add_argument('--traits', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu', help="e.g. 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X, genotypes, environment = sorghum.load()
    train_test = (sorghum.folds(genotypes) == args.fold).astype(int)

    model = TraitModels(args.traits, LinearModel, X.shape[1])
    train(model, X, genotypes, environment, train_test, model_file=args.out_model,
          n_traits=args.traits, n_iter=args.steps, learning_rate=1e-5, noise_level=0.005,
          penalty=smoothness_penalty(1e-4), device=args.device)
    torch.save(model, args.out_model)

    with torch.no_grad():
        traits = model(torch.tensor(X).float().to(args.device)).cpu().numpy()
    np.savez(args.out_traits, traits=traits)


if __name__ == '__main__':
    main()
