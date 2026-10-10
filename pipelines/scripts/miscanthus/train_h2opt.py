"""Train Miscanthus linear H2-opt on one date, one model per plant fold.

h2opt.train_models: h2opt.LinearModel on all pixels of the date (h2opt_analysis.miscanthus.
load_date, flattened), Henderson3 heritability (families, plots within families, rep and field
column fixed), Adam, normal input noise of the standard deviation of choose_noise.py for the fold
and date, each fold's model on its training plants. Writes the fold models (a list, torch.save) to
--out-models and an .npz with 'traits' (folds x plants x traits, uncorrelated on each fold's
training plants).

    python pipelines/scripts/miscanthus/train_h2opt.py --population central --images images.npy \\
        --noise noise.npz --date 0 --traits 1 --steps 10000 --device cuda --out date0.npz \\
        --out-models date0.pt
"""

import argparse

import numpy as np
import torch
from h2opt import LinearModel, TraitModels, synthetic_traits, train_models

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--noise', required=True, help='choose_noise.py output')
    parser.add_argument('--date', type=int, required=True, help='index into miscanthus.DATES')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    parser.add_argument('--out', required=True)
    parser.add_argument('--out-models', required=True, help='.pt: the list of fold models')
    args = parser.parse_args()

    torch.manual_seed(args.date)
    design = miscanthus.design(args.population)
    x = miscanthus.load_date(args.images, design.keep, args.date)
    x = x.reshape((len(x), -1))
    with np.load(args.noise) as f:
        noise_sd = f['noise_sd'][:, args.date]
    folds = np.arange(miscanthus.N_FOLDS)
    train_test = np.array([(design.fold == f).astype(int) for f in folds])
    models = [TraitModels(args.traits, LinearModel, x.shape[1]) for _ in folds]
    train_models(models, x, design.families, design.environment, train_test,
                 n_traits=args.traits, n_iter=args.steps, noise_sd=noise_sd[:, None],
                 device=args.device, subgroups=design.plots, estimator='henderson3')
    torch.save(models, args.out_models)
    np.savez(args.out, traits=np.stack([synthetic_traits(m, x, tt == 0)
                                        for m, tt in zip(models, train_test, strict=True)]))


if __name__ == '__main__':
    main()
