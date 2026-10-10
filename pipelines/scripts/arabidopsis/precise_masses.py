"""Compact table of the precise masses of one Arabidopsis treatment, for enrichment_inputs.py.

Streams arabidopsis.precise_masses(treatment) (36.6 GB as float64; never loaded whole) and keeps
the m/z and retention-time bins of the features (arabidopsis.N_MZ, N_RT) with nonzero intensity.
Per kept entry: the m/z and retention-time bin, the precise m/z (bin - 0.5 + step / 10000), the
intensity-weighted mean retention time (s) and the standard deviation of the intensity over the
treatment's samples. Writes an .npz.

    python pipelines/scripts/arabidopsis/precise_masses.py --treatment 0d --out precise_0d.npz
"""

import argparse
import zipfile

import numpy as np

from h2opt_analysis import arabidopsis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    rows, _, _ = arabidopsis.design(args.treatment)
    n_samples = rows.sum()
    with zipfile.ZipFile(arabidopsis.precise_masses(args.treatment)) as archive:
        member = archive.open(archive.namelist()[0])
        version = np.lib.format.read_magic(member)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0) else
                                 np.lib.format.read_array_header_2_0)(member)
        assert not fortran and len(shape) == 4 and shape[0] == 4, shape
        _, n_mz, n_rt, n_steps = shape
        block = n_rt * n_steps * dtype.itemsize
        sums = np.zeros((3, arabidopsis.N_MZ, arabidopsis.N_RT, n_steps), np.float64)
        for stat in range(4):
            for mz in range(n_mz):
                values = np.frombuffer(member.read(block), dtype).reshape(n_rt, n_steps)
                if stat != 2 and mz < arabidopsis.N_MZ:
                    sums[[0, 1, None, 2][stat], mz] = values[:arabidopsis.N_RT]
    total, weighted_rt, squares = sums
    keep = total > 0
    mz_bin, rt_bin, step = np.nonzero(keep)
    mean = total[keep] / n_samples
    sd = np.sqrt(np.maximum(squares[keep] / n_samples - mean ** 2, 0))
    np.savez(args.out, mz_bin=mz_bin.astype(np.int16), rt_bin=rt_bin.astype(np.int16),
             mz=mz_bin - 0.5 + step / 10000, rt=weighted_rt[keep] / total[keep], sd=sd)
    print(f'{args.treatment}: {n_samples} samples, {keep.sum()} entries with intensity')


if __name__ == '__main__':
    main()
