"""The noise levels of H2-opt on one image simulation instance: fixed for every instance (PI), not
chosen by selection. Writes the linear level in the format of choose_noise.py (.npz with 'chosen'
and 'noise_sd' = the level x h2opt.selection.noise_scale of the training images), read by
train_h2opt.py and train_hybrid.py, and the CNN level in the format of choose_cnn_level.py (.npz
with 'chosen'), read by train_hybrid.py.

    python pipelines/scripts/image_simulation/fixed_levels.py --measurements images.npz \\
        --linear-level 17.7828 --cnn-level 1 --out-noise noise.npz --out-cnn-level cnn_level.npz
"""

import argparse

import numpy as np
from h2opt.selection import noise_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', required=True, help='.npz with X, names, test')
    parser.add_argument('--linear-level', type=float, required=True)
    parser.add_argument('--cnn-level', type=float, required=True)
    parser.add_argument('--out-noise', required=True)
    parser.add_argument('--out-cnn-level', required=True)
    args = parser.parse_args()
    with np.load(args.measurements) as f:
        X, test = f['X'], f['test']
    noise_sd = args.linear_level * noise_scale(X[~test].reshape(((~test).sum(), -1)))
    np.savez(args.out_noise, chosen=args.linear_level, noise_sd=noise_sd)
    np.savez(args.out_cnn_level, chosen=args.cnn_level)
    print(f'linear level {args.linear_level:g} (noise SD {noise_sd:.4g}), CNN level '
          f'{args.cnn_level:g} (fixed)', flush=True)


if __name__ == '__main__':
    main()
