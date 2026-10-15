"""Encode the true traits of all simulation instances into simulated spectra.

One random stream (seed 0) runs over the instances in order, so all instances are made in one
run, and --out-dirs names one directory per instance. Writes, per instance, spectra.npz (X:
spectra, names: line of each plant, lines: .fam row of each line, test: test plants, dimensions:
latent dimension of each trait) and truth.npz (traits: the true traits).

    python pipelines/scripts/simulation/encode.py --autoencoder autoencoder.pt \\
        --reference X.npz --out-dirs instance0 ... instance9
"""

import argparse
from pathlib import Path

import numpy as np
from h2opt import load_npz

from h2opt_analysis import simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoencoder', required=True, help='pickled autoencoder (.pt)')
    parser.add_argument('--reference', required=True, help='real spectra (.npz, arr_0)')
    parser.add_argument('--out-dirs', nargs='+', required=True,
                        help='output directory of each instance, in instance order')
    args = parser.parse_args()
    if len(args.out_dirs) != simulation.N_INSTANCES:
        parser.error(f'give {simulation.N_INSTANCES} output directories, one per instance')

    autoencoder = simulation.load_paper_autoencoder(args.autoencoder)
    reference = load_npz(args.reference)
    names, truths = zip(*(simulation.load_truth(i) for i in range(simulation.N_INSTANCES)),
                        strict=True)
    encoded = simulation.encode_instances(truths, autoencoder, reference)
    tests = simulation.test_lines(names)

    for i, out in enumerate(args.out_dirs):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        X, dimensions = encoded[i]
        np.savez_compressed(out / 'spectra.npz', X=X, names=names[i], test=tests[i],
                            lines=simulation.line_index(i, names[i]), dimensions=dimensions)
        np.savez_compressed(out / 'truth.npz', traits=truths[i])
        print(f'instance {i}: {X.shape[0]} plants, {tests[i].mean():.3f} test, '
              f'trait dimensions {dimensions}', flush=True)


if __name__ == '__main__':
    main()
