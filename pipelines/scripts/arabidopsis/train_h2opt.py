"""Train linear H2-opt on one Arabidopsis treatment for the accession folds, all folds at once.

One linear model per trait and fold, and normal input noise with the standard deviation that
choose_noise.py chose for that fold (one for all traits). An L1 penalty (--penalty l1 1e-6, from
the config) but not the paper's clipping, so the loss is the heritability that the evaluation
scores plus the penalty. Trained with h2opt.train's
default optimizer (Adam), all folds together by h2opt.train_models. Writes, per fold, the model
(.pt) and the traits of all samples of the treatment, uncorrelated on the training samples (.npz,
array 'traits').

    python pipelines/scripts/arabidopsis/train_h2opt.py --features features.npy --treatment 6d \\
        --folds 0 1 --noise noise.npz --out-traits fold0.npz fold1.npz \\
        --out-models fold0.pt fold1.pt --steps 16000 --traits 10 --seed 0 --penalty l1 1e-6 \\
        --device cuda
"""

import argparse

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, synthetic_traits, train_models

from h2opt_analysis import arabidopsis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
    parser.add_argument('--folds', type=int, nargs='+', required=True)
    parser.add_argument('--noise', required=True, help='the noise chosen by choose_noise.py')
    parser.add_argument('--out-traits', nargs='+', required=True, help='one file per fold')
    parser.add_argument('--out-models', nargs='+', required=True, help='one file per fold')
    parser.add_argument('--steps', type=int, required=True, help='steps per trait')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--penalty', nargs=2, required=True, metavar=('KIND', 'WEIGHT'),
                        help="penalty of h2opt.train: 'l1', 'l2' or 'tv', and its weight")
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()
    penalty = (args.penalty[0], float(args.penalty[1]))
    if not len(args.folds) == len(args.out_traits) == len(args.out_models):
        parser.error('give one traits file and one model file per fold')

    torch.manual_seed(args.seed)
    X, accessions, environment = arabidopsis.load(args.features, args.treatment)
    fold_of = arabidopsis.folds(accessions)
    with np.load(args.noise) as f:
        noise_folds, noise_sd = list(f['folds']), f['noise_sd']
    noise_sd = noise_sd[[noise_folds.index(fold) for fold in args.folds], None]

    models = [TraitModels(args.traits, LinearModel, X.shape[1]) for _ in args.folds]
    train_test = np.array([(fold_of == fold).astype(int) for fold in args.folds])
    train_models(models, X, accessions, environment, train_test, model_files=args.out_models,
                 n_traits=args.traits, n_iter=args.steps, noise_sd=noise_sd, penalty=penalty,
                 device=args.device)
    for model, rows, path in zip(models, train_test, args.out_traits, strict=True):
        np.savez(path, traits=synthetic_traits(model, X, rows == 0))


if __name__ == '__main__':
    main()
