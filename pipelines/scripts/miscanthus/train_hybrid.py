"""Train Miscanthus hybrid H2-opt on one date, one model per plant fold.

h2opt.train_linear_conv_models: linear H2-opt on all pixels of the date (the noise of
choose_noise.py for the fold and date, linear_steps steps), then h2opt.LinearConvModel, the linear
map plus h2opt.PositionImageConvModel (a weight per position: the plant is at the center, its
neighbors at the edges), trained together (steps steps). The linear branch has the linear level of
choose_noise.py; the CNN branch has the CNN level of choose_cnn_level.py, times noise_scale of the
fold's training pixels of the date. Both levels serve all folds and dates.
Henderson3 heritability (families, plots within families, rep and field column fixed). Writes the
fold models (a list, torch.save) to --out-models and an .npz with 'traits' (folds x plants x
traits, uncorrelated on each fold's training plants). With --earlier (the .npz files of earlier
runs, e.g. traits 1 and 2), the new traits are trained on top of each fold's earlier traits, all
files together (the loss uses the
residual after regression on them, h2opt.train_models), and saved as that residual on the fold's
training plants, standardized there.

    python pipelines/scripts/miscanthus/train_hybrid.py --population central --images images.npy \\
        --noise noise.npz --cnn-level cnn_level.npz --date 0 --traits 1 --linear-steps 10000 \\
        --steps 1000 --max-copies 4 --device cuda --out date0.npz \\
        --out-models date0.pt
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
    parser.add_argument('--cnn-level', required=True, help='choose_cnn_level.py output')
    parser.add_argument('--date', type=int, required=True, help='index into miscanthus.DATES')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--linear-steps', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max-copies', type=int,
                        help='models trained at once (GPU memory)')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    parser.add_argument('--out', required=True)
    parser.add_argument('--out-models', required=True, help='.pt: the list of fold models')
    parser.add_argument('--neighbor-width', type=int,
                        help='train on the heritability after the nearest-neighbor field '
                             'adjustment (h2opt.spatial), neighbors within this many columns')
    parser.add_argument('--earlier', nargs='+',
                        help='.npz files of this script: train on top of their traits')
    args = parser.parse_args()

    torch.manual_seed(args.date)
    design = miscanthus.design(args.population)
    x = miscanthus.load_date(args.images, design.keep, args.date)
    flat = x.reshape((len(x), -1))
    with np.load(args.noise) as f:
        noise_sd = f['noise_sd'][:, args.date]
    with np.load(args.cnn_level) as f:
        level = float(f['chosen'])
    folds = np.arange(miscanthus.N_FOLDS)
    earlier = None if args.earlier is None else np.concatenate(
        [np.load(path)['traits'] for path in args.earlier], axis=-1)
    train_test = np.array([(design.fold == f).astype(int) for f in folds])
    conv_sd = np.array([level * noise_scale(flat[design.fold != f]) for f in folds])
    models = train_linear_conv_models(
        lambda: PositionImageConvModel(x.shape[1], x.shape[2:]), x, design.families,
        design.environment, train_test, noise_sd[:, None], conv_sd[:, None], n_traits=args.traits,
        n_linear_iter=args.linear_steps, n_iter=args.steps, device=args.device,
        subgroups=design.plots, estimator='henderson3', max_copies=args.max_copies,
        neighbors=None if args.neighbor_width is None
        else miscanthus.neighbor_weights(design, args.neighbor_width),
        earlier=earlier)
    torch.save(models, args.out_models)
    traits = []
    for f, (model, tt) in enumerate(zip(models, train_test, strict=True)):
        Y = synthetic_traits(model, x, tt == 0)
        if earlier is not None:
            train = tt == 0
            basis = np.column_stack([np.ones(len(Y)), earlier[f]])
            Y = Y - basis @ np.linalg.lstsq(basis[train], Y[train], rcond=None)[0]
            Y = (Y - Y[train].mean(axis=0)) / Y[train].std(axis=0)
        traits.append(Y)
    np.savez(args.out, traits=np.stack(traits))


if __name__ == '__main__':
    main()
