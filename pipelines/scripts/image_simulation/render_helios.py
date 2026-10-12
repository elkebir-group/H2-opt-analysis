"""Render the plot images of one image-simulation instance with Helios
(h2opt_analysis.helios_simulation). Writes the images (.npz with X, names, test) and the true
traits (.npz with traits).

    python pipelines/scripts/image_simulation/render_helios.py --instance 0 \\
        --binary helios/build/field_render --workdir helios_tiles/instance0 \\
        --out-images images.npz --out-truth truth.npz
"""

import argparse

import numpy as np

from h2opt_analysis import helios_simulation, miscanthus, simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--out-images', required=True)
    parser.add_argument('--out-truth', required=True)
    args = parser.parse_args()

    names, truth = simulation.load_truth(args.instance)
    tests = simulation.test_lines([simulation.load_truth(i)[0]
                                   for i in range(simulation.N_INSTANCES)])
    z = (truth - truth.mean(axis=0)) / truth.std(axis=0)
    X = helios_simulation.render(z[:, 1], z[:, 2], z[:, 0], args.instance, args.binary,
                                 args.workdir)
    np.savez(args.out_images, X=miscanthus.scale_images(X), names=names, test=tests[args.instance])
    np.savez(args.out_truth, traits=truth)
    print(f'instance {args.instance}: images {X.shape}, {tests[args.instance].mean():.3f} of the '
          'plants are test')


if __name__ == '__main__':
    main()
