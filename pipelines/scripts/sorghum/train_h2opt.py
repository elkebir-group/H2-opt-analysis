"""Train linear H2-opt on sorghum for one of the paper's 10 genotype folds.

One linear model per trait, normal input noise with the standard deviation chosen by
choose_noise.py, and no clipping of the traits in the loss, so the loss is the heritability that
the evaluation scores (the paper clipped at 2). Trained with h2opt.train's default optimizer
(Adam). Writes the model (.pt) and the traits of all plants,
uncorrelated on the training plants (.npz, array 'traits').

    python pipelines/scripts/sorghum/train_h2opt.py --fold 0 --noise noise.npz \\
        --out-traits fold0.npz --out-model fold0.pt --steps 10000 --traits 10 --seed 0 \\
        --device cuda
"""

import argparse

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, synthetic_traits, train

from h2opt_analysis import sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--noise', required=True, help='the noise chosen by choose_noise.py')
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--out-model', required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per trait')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X, genotypes, environment = sorghum.load()
    is_train = sorghum.folds(genotypes) != args.fold
    with np.load(args.noise) as f:
        noise_sd = float(f['noise_sd'])

    model = TraitModels(args.traits, LinearModel, X.shape[1])
    train(model, X, genotypes, environment, (~is_train).astype(int), model_file=args.out_model,
          n_traits=args.traits, n_iter=args.steps, noise_level=noise_sd, noise='normal',
          clip=None, device=args.device)
    np.savez(args.out_traits, traits=synthetic_traits(model, X, is_train))


if __name__ == '__main__':
    main()
