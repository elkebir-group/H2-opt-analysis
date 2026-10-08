"""Train the CNN/linear hybrid H2-opt on one simulation instance.

h2opt.train_linear_conv_models: linear H2-opt (one linear model per trait), then
h2opt.LinearConvModel, the linear map plus h2opt.ConvModel over wavelength, warm started from each
linear trait. The linear stage and the linear branch use normal noise of the standard deviation
of linear H2-opt (choose_noise.py); the CNN branch gets normal input noise of standard deviation
the level of choose_cnn_level.py x h2opt.selection.noise_scale of the training spectra. No
clipping, Adam (h2opt.train's default). Writes the model (.pt) and the traits of all plants,
uncorrelated on the training plants (.npz, array 'traits').

    python pipelines/scripts/simulation/train_hybrid.py --spectra spectra.npz --noise noise.npz \\
        --cnn-level cnn_level.npz --out-traits h2opt_hybrid.npz --out-model h2opt_hybrid.pt \\
        --traits 5 --linear-steps 10000 --steps 1000 --seed 0 --device cuda
"""

import argparse
import functools

import numpy as np
import torch
from h2opt import ConvModel, synthetic_traits, train_linear_conv_models
from h2opt.selection import noise_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--noise', required=True, help='choose_noise.py output')
    parser.add_argument('--cnn-level', required=True, help='choose_cnn_level.py output')
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
    with np.load(args.noise) as f:
        noise_sd = float(f['noise_sd'])
    with np.load(args.cnn_level) as f:
        cnn_level = float(f['chosen'])

    model, = train_linear_conv_models(
        functools.partial(ConvModel, X.shape[1]), X, names, None, test.astype(int)[None],
        noise_sd, cnn_level * noise_scale(X[~test]), n_traits=args.traits,
        n_linear_iter=args.linear_steps, n_iter=args.steps, device=args.device)
    torch.save(model, args.out_model)
    np.savez(args.out_traits, traits=synthetic_traits(model, X, ~test))


if __name__ == '__main__':
    main()
