"""Measurements of all Arabidopsis samples: cropped and scaled, flattened.

h2opt_analysis.arabidopsis.features() (m/z 0-1499, retention time 0-20 min, MS1 and MS/MS, over
the largest intensity), saved as float32 (1112 x 120,000) in sample order.

    python pipelines/scripts/arabidopsis/features.py --out features.npy
"""

import argparse

import numpy as np

from h2opt_analysis import arabidopsis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    X = arabidopsis.features()
    np.save(args.out, X.astype(np.float32))
    print(f'{X.shape[0]} samples x {X.shape[1]} features')


if __name__ == '__main__':
    main()
