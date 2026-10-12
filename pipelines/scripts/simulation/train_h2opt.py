"""Train linear H2-opt on one simulation instance.

One linear model per trait, normal input noise of the standard deviation chosen by
choose_noise.py, and no clipping (the paper's simulation setting).
Trained with h2opt.train's default optimizer (Adam) on the training plants (outside fold 0 of
simulation.line_folds, the test plants).
Writes the model (.pt) and the traits of all plants, uncorrelated on the training plants (.npz,
array 'traits').

    python pipelines/scripts/simulation/train_h2opt.py --measurements spectra.npz \\
        --noise noise.npz \\
        --out-traits h2opt.npz --out-model h2opt.pt --traits 5 --steps 10000 \\
        --seed 0 --instance 0 --device cuda
"""

import argparse

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, synthetic_traits, train

from h2opt_analysis import simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', required=True,
                        help='.npz with X (plants first; images are flattened), names, test')
    parser.add_argument('--noise', required=True, help='choose_noise.py output')
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--out-model', required=True)
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per trait')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with np.load(args.measurements) as f:
        X, names = f['X'], f['names']
    test = simulation.instance_folds(args.instance) == 0
    X = X.reshape((len(X), -1))
    with np.load(args.noise) as f:
        noise_sd = float(f['noise_sd'])

    model = TraitModels(args.traits, LinearModel, X.shape[1])
    train(model, X, names, None, test.astype(int), model_file=args.out_model,
          n_traits=args.traits, n_iter=args.steps, noise_sd=noise_sd,
          device=args.device)
    np.savez(args.out_traits, traits=synthetic_traits(model, X, ~test))


if __name__ == '__main__':
    main()
