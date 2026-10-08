"""Train the CNN/linear hybrid H2-opt on one simulation instance.

h2opt.train_linear_conv_models: linear H2-opt (one linear model per trait), then
h2opt.LinearConvModel, the linear map plus h2opt.ConvModel over wavelength, warm started from each
linear trait. Both stages and both branches use normal input noise of one fixed standard
deviation (the config's simulation.h2opt.noise_sd, the noise of linear H2-opt), so the linear
branch gets no extra noise. No clipping, Adam (h2opt.train's default). Writes the model (.pt) and
the traits of all plants, uncorrelated on the training plants (.npz, array 'traits').

    python pipelines/scripts/simulation/train_hybrid.py --spectra spectra.npz --noise-sd 0.02 \\
        --out-traits h2opt_hybrid.npz --out-model h2opt_hybrid.pt --traits 5 --linear-steps 10000 \\
        --steps 1000 --seed 0 --device cuda
"""

import argparse
import functools

import numpy as np
import torch
from h2opt import ConvModel, synthetic_traits, train_linear_conv_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--noise-sd', type=float, required=True,
                        help='standard deviation of the input noise')
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--out-model', required=True)
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--linear-steps', type=int, required=True,
                        help='steps per trait of the linear stage')
    parser.add_argument('--steps', type=int, required=True, help='steps per trait of the hybrid')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with np.load(args.spectra) as f:
        X, names, test = f['X'], f['names'], f['test']

    model, = train_linear_conv_models(
        functools.partial(ConvModel, X.shape[1]), X, names, None, test.astype(int)[None],
        args.noise_sd, args.noise_sd, n_traits=args.traits, n_linear_iter=args.linear_steps,
        n_iter=args.steps, device=args.device)
    torch.save(model, args.out_model)
    np.savez(args.out_traits, traits=synthetic_traits(model, X, ~test))


if __name__ == '__main__':
    main()
