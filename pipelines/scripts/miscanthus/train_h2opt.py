"""Train Miscanthus H2-opt (the image CNN) on one date, all folds at once.

Trait 1 of h2opt.ImageConvModel per fold, trained on the fold's training plants with Henderson3
heritability (families, plots within families, rep and field column fixed) and normal input noise
of the standard deviation that choose_noise.py chose for the fold and date (h2opt.train_models).
Writes the traits of all plants per fold, uncorrelated on the fold's training plants (.npz, array
'traits': folds x plants x 1), and the models (.pt, one TraitModels per fold).

    python pipelines/scripts/miscanthus/train_h2opt.py --population central --images images.npy \\
        --noise noise.npz --date 2 --out-traits date2.npz --out-models date2.pt --steps 1000 \\
        --seed 0 --max-copies 6 --device cuda
"""

import argparse

import numpy as np
import torch
from h2opt import ImageConvModel, TraitModels, synthetic_traits, train_models

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--noise', required=True, help='the noise chosen by choose_noise.py')
    parser.add_argument('--date', type=int, required=True, help='index into DATES')
    parser.add_argument('--out-traits', required=True)
    parser.add_argument('--out-models', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-copies', type=int, required=True, help='models trained at once')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    design = miscanthus.design(args.population)
    images = miscanthus.load_date(args.images, design.keep, args.date)
    with np.load(args.noise) as f:
        folds, noise_sd = f['folds'], f['noise_sd'][:, args.date]
    models = [TraitModels(1, ImageConvModel, images.shape[1], images.shape[2:]) for _ in folds]
    train_test = np.array([(design.fold == fold).astype(int) for fold in folds])
    train_models(models, images, design.families, design.environment, train_test, n_traits=1,
                 n_iter=args.steps, noise_sd=noise_sd[:, None],
                 device=args.device, subgroups=design.plots, estimator='henderson3',
                 max_copies=args.max_copies)
    traits = np.stack([synthetic_traits(model, images, rows == 0)
                       for model, rows in zip(models, train_test, strict=True)])
    np.savez(args.out_traits, traits=traits)
    torch.save([model.cpu() for model in models], args.out_models)


if __name__ == '__main__':
    main()
