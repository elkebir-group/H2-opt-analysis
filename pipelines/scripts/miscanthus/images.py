"""Write the Miscanthus images of one population as an uncompressed float32 .npy.

The images arrive as a compressed float64 .npz (h2opt_analysis.miscanthus.image_archive), which
must be decompressed whole to read one date. The .npy (3900 plants x 14 dates x 6 channels x
63 x 63) can be memory-mapped, so each job reads only the dates it needs.

    python pipelines/scripts/miscanthus/images.py --population central --out images.npy
"""

import argparse

import numpy as np

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with np.load(miscanthus.image_archive(args.population)) as archive:
        images = archive['arr_0'].astype(np.float32)
    print('images', images.shape, flush=True)
    np.save(args.out, images)


if __name__ == '__main__':
    main()
