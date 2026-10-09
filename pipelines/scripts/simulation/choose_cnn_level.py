"""Choose the noise level of the CNN branch of the hybrid H2-opt on one simulation instance.

h2opt.selection.score_linear_conv_levels on the training plants: 20 validation splits of the
training lines (4 partitions into 5 folds); for each split and CNN level of
h2opt.selection.NOISE_LEVELS, the first trait of linear H2-opt (the noise SD of choose_noise.py,
linear_steps steps), then the CNN branch (simulation.make_cnn: h2opt.ConvModel over wavelength, or
h2opt.ImageConvModel for images; steps steps) with input noise of standard deviation level x
noise_scale of the training measurements. The chosen level is the highest level whose held-out
heritability of trait 1 is within one standard error of the best
(h2opt.selection.highest_level_within_one_se, the rule of every level): a high level keeps the
hybrid near the linear trait, so the CNN must clearly raise the held-out heritability to get a low
level. Writes an .npz (levels; chosen; scores: splits x levels) read by train_hybrid.py.

    python pipelines/scripts/simulation/choose_cnn_level.py --measurements spectra.npz \\
        --noise noise.npz --out cnn_level.npz \\
        --linear-steps 10000 --steps 1000 --seed 0 --device cuda
"""

import argparse

import numpy as np
from h2opt.selection import NOISE_LEVELS, highest_level_within_one_se, score_linear_conv_levels

from h2opt_analysis import simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', required=True,
                        help='.npz with X (plants x features, or plants x channels x height x '
                             'width), names, test')
    parser.add_argument('--noise', required=True, help='choose_noise.py output')
    parser.add_argument('--out', required=True)
    parser.add_argument('--linear-steps', type=int, required=True,
                        help='steps of the linear stage')
    parser.add_argument('--steps', type=int, required=True, help='steps of the CNN branch')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-copies', type=int,
                        help='models trained at once (GPU memory only; same result)')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    args = parser.parse_args()

    with np.load(args.measurements) as f:
        X, names, test = f['X'], f['names'], f['test']
    with np.load(args.noise) as f:
        noise_sd = float(f['noise_sd'])
    levels = NOISE_LEVELS
    scores, _ = score_linear_conv_levels(
        simulation.make_cnn(X), X, names, None, noise_sd, levels,
        subsets=(~test)[None], seed=args.seed, n_linear_iter=args.linear_steps,
        n_iter=args.steps, device=args.device, max_copies=args.max_copies)
    chosen = highest_level_within_one_se(levels, scores[0])
    np.savez(args.out, levels=levels, chosen=chosen, scores=scores[0])
    mean = scores[0].mean(axis=0)
    se = scores[0].std(axis=0, ddof=1) / np.sqrt(len(scores[0]))
    print(f'chosen CNN level {chosen:g}; held-out heritability of trait 1 per level '
          '(mean over splits, SE):')
    for level, m, s in zip(levels, mean, se, strict=True):
        print(f'  {level:<9.3g} {m:.4f} ({s:.4f})')


if __name__ == '__main__':
    main()
