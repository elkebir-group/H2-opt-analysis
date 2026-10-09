"""Train Miscanthus hybrid H2-opt on one date, one model per plant fold.

h2opt.train_linear_conv_models: linear H2-opt on all pixels of the date (the noise of
choose_noise.py for the fold and date, linear_steps steps), then h2opt.LinearConvModel, the linear
map plus h2opt.PositionImageConvModel (a weight per position: the plant is at the center, its
neighbors at the edges), trained together (steps steps). The CNN branch has the noise level of
choose_noise.py too (the one level), times noise_scale of the fold's training pixels of the date.
Henderson3 heritability (families, plots within families, rep and field column fixed). Writes an
.npz with 'traits' (folds x plants x traits, uncorrelated on each fold's training plants).

    python pipelines/scripts/miscanthus/train_hybrid.py --population central --images images.npy \\
        --noise noise.npz --date 0 --traits 1 --linear-steps 10000 \\
        --steps 1000 --max-copies 4 --device cuda --out date0.npz
"""

import argparse

import numpy as np
import torch
from h2opt import PositionImageConvModel, synthetic_traits, train_linear_conv_models
from h2opt.selection import noise_scale

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--noise', required=True, help='choose_noise.py output')
    parser.add_argument('--date', type=int, required=True, help='index into miscanthus.DATES')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--linear-steps', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max-copies', type=int,
                        help='models trained at once (GPU memory only; same result)')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    torch.manual_seed(args.date)
    design = miscanthus.design(args.population)
    x = miscanthus.load_date(args.images, design.keep, args.date)
    flat = x.reshape((len(x), -1))
    with np.load(args.noise) as f:
        noise_sd, level = f['noise_sd'][:, args.date], float(f['chosen'])
    folds = np.arange(miscanthus.N_FOLDS)
    train_test = np.array([(design.fold == f).astype(int) for f in folds])
    conv_sd = np.array([level * noise_scale(flat[design.fold != f]) for f in folds])
    models = train_linear_conv_models(
        lambda: PositionImageConvModel(x.shape[1], x.shape[2:]), x, design.families,
        design.environment, train_test, noise_sd[:, None], conv_sd[:, None], n_traits=args.traits,
        n_linear_iter=args.linear_steps, n_iter=args.steps, device=args.device,
        subgroups=design.plots, estimator='henderson3', max_copies=args.max_copies)
    np.savez(args.out, traits=np.stack([synthetic_traits(m, x, tt == 0)
                                        for m, tt in zip(models, train_test, strict=True)]))


if __name__ == '__main__':
    main()
