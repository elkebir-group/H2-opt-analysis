"""Train linear H2-opt on one simulation instance with the paper's simulation settings.

5 traits, one linear model per trait, RMSprop with learning rate 1e-5 for 10,000 steps per trait,
Gaussian input noise with standard deviation 0.02 and no clipping. Writes the model (.pt) and the
traits of all plants, decorrelated over all plants (.npz, array 'traits').

    python pipelines/scripts/simulation/train_h2opt.py --spectra spectra.npz \\
        --out-traits h2opt_rmsprop.npz --out-model h2opt_rmsprop.pt [--device cuda] [--seed 0]
"""

import argparse

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, decorrelate, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--out-model', required=True)
    parser.add_argument('--traits', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10000, help='RMSprop steps per trait')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu', help="e.g. 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with np.load(args.spectra) as f:
        X, names, test = f['X'], f['names'], f['test']

    model = TraitModels(args.traits, LinearModel, X.shape[1])
    train(model, X, names, None, test.astype(int), model_file=args.out_model,
          n_traits=args.traits, n_iter=args.steps, learning_rate=1e-5, noise_level=0.02,
          noise='normal', clip=None, device=args.device)
    torch.save(model, args.out_model)

    with torch.no_grad():
        traits = decorrelate(model(torch.tensor(X).float().to(args.device))).cpu().numpy()
    np.savez(args.out_traits, traits=traits)


if __name__ == '__main__':
    main()
