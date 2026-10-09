"""Simulate the plot images of one image-simulation instance (h2opt_analysis.image_simulation).

Writes the images (.npz: X, plants x 6 x 63 x 63 float32; names, the line of each plant; test, the
test plants) and the true traits (.npz: traits, plants x 3).

    python pipelines/scripts/image_simulation/simulate.py --instance 0 --out-images images.npz \\
        --out-truth truth.npz
"""

import argparse

import numpy as np

from h2opt_analysis import image_simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--out-images', required=True)
    parser.add_argument('--out-truth', required=True)
    args = parser.parse_args()

    X, names, test, truth = image_simulation.make_instance(args.instance)
    np.savez(args.out_images, X=X, names=names, test=test)
    np.savez(args.out_truth, traits=truth)
    print(f'instance {args.instance}: images {X.shape}, {test.mean():.3f} of the plants are test')


if __name__ == '__main__':
    main()
