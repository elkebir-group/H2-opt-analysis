"""Train linear H2-opt on sorghum for one of the paper's 10 genotype folds (the paper's settings).

10 traits, one linear model per trait, RMSprop with learning rate 1e-5 for 10,000 steps per trait, uniform input
noise up to 0.005, and a penalty of 1e-4 on the total variation of the weights across adjacent wavelengths.
Writes results/sorghum/h2opt/fold{k}.pt (model) and fold{k}.npz (traits of all plants).

    uv run python scripts/sorghum/train_h2opt.py --fold 0 [--steps 10000] [--traits 10]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, train

from h2opt_analysis import sorghum

RESULTS = Path('results/sorghum/h2opt')


def smoothness_penalty(scale):
    """Total variation of the weights across adjacent wavelengths, relative to the spread of the trait."""
    def penalty(trait_model, Y):
        w = trait_model.lin1.weight
        variation = torch.mean(torch.sum(torch.abs(w[:, 1:] - w[:, :-1]), axis=1))
        return scale * variation / torch.mean(torch.abs(Y - torch.mean(Y, axis=0)))
    return penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--traits', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X, genotypes, environment = sorghum.load()
    train_test = (sorghum.folds(genotypes) == args.fold).astype(int)

    RESULTS.mkdir(parents=True, exist_ok=True)
    model_file = RESULTS / f'fold{args.fold}.pt'
    model = TraitModels(args.traits, LinearModel, X.shape[1])
    train(model, X, genotypes, environment, train_test, model_file=model_file, n_traits=args.traits,
          n_iter=args.steps, learning_rate=1e-5, noise_level=0.005, penalty=smoothness_penalty(1e-4))
    torch.save(model, model_file)

    with torch.no_grad():
        traits = model(torch.tensor(X).float()).numpy()
    np.savez(RESULTS / f'fold{args.fold}.npz', traits=traits)


if __name__ == '__main__':
    main()
