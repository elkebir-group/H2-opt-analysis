"""Choose the noise level of linear H2-opt on one simulation instance, one for all traits.

h2opt.selection.select_noise_level on the training plants: 5 validation splits of the training
lines, every level of h2opt.selection.NOISE_LEVELS, the first trait trained with the steps of
train_h2opt.py; the level whose first trait has the best held-out heritability averaged over the
splits. The test plants never take part in the choice. Writes an .npz (levels; chosen; noise_sd:
the chosen level times noise_scale of the training spectra; scores: splits x levels) read by
train_h2opt.py, choose_cnn_level.py, train_hybrid.py, baseline.py and plot_noise_sweep.py.

    python pipelines/scripts/simulation/choose_noise.py --spectra spectra.npz --out noise.npz \\
        --steps 10000 --seed 0 --device cuda
"""

import argparse

import numpy as np
from h2opt import LinearModel
from h2opt.selection import NOISE_LEVELS, best_level, select_noise_level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per training run')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    with np.load(args.spectra) as f:
        X, names, test = f['X'], f['names'], f['test']
    noise_sd, scores, _ = select_noise_level(lambda: LinearModel(X.shape[1]), X, names, None,
                                             subsets=(~test)[None], seed=args.seed,
                                             n_iter=args.steps, device=args.device)
    chosen = best_level(NOISE_LEVELS, scores[0])
    np.savez(args.out, levels=NOISE_LEVELS, chosen=chosen, noise_sd=noise_sd[0],
             scores=scores[0])
    print(f'chosen level {chosen:g} (noise SD {noise_sd[0]:.4g}); held-out heritability of '
          'trait 1 per level (mean over splits):')
    for level, score in zip(NOISE_LEVELS, scores[0].mean(axis=0), strict=True):
        print(f'  {level:<9.3g} {score:.4f}')


if __name__ == '__main__':
    main()
