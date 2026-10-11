"""The CNN level of the hybrid on one image simulation instance: one fixed level for every instance
(PI), not chosen by selection. Writes it in the format of choose_cnn_level.py (.npz with 'chosen'),
which train_hybrid.py reads.

    python pipelines/scripts/image_simulation/fixed_cnn_level.py --level 0.316228 \\
        --out cnn_level.npz
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=float, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    np.savez(args.out, chosen=args.level)
    print(f'CNN level {args.level:g} (fixed)', flush=True)


if __name__ == '__main__':
    main()
