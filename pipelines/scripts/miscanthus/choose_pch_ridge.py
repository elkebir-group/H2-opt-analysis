"""Choose the one PCH ridge of Miscanthus: all folds, all dates.

PCH.tune on the training plants of each fold, on one date (--date): PCH on all pixels of the date
(h2opt_analysis.miscanthus.load_date, flattened), Henderson3 heritability (families, plots within
families, rep and field column fixed), 20 validation splits of the training plants (4 partitions
into 5 folds), every ridge of h2opt.baselines.RIDGES. The rule
h2opt.selection.highest_level_within_one_se takes the 100 validation splits of all folds together
and gives one ridge, the same for all folds and dates: the procedure of the linear H2-opt level
(choose_noise.py). Writes an .npz (date; folds; ridges; chosen: the ridge; scores: folds x splits
x ridges).

    python pipelines/scripts/miscanthus/choose_pch_ridge.py --population central \\
        --images images.npy --date 2 --out ridge.npz
"""

import argparse

import numpy as np
from h2opt.baselines import PCH, RIDGES
from h2opt.selection import highest_level_within_one_se

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--date', type=int, required=True,
                        help='index into miscanthus.DATES of the date of the search')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    folds = np.arange(miscanthus.N_FOLDS)
    x = miscanthus.load_date(args.images, design.keep, args.date)
    x = x.reshape((len(x), -1))
    scores = []
    for f in folds:
        rows = design.fold != f
        scores.append(PCH.tune(
            x[rows], design.families[rows], design.environment[rows], subgroups=design.plots[rows],
            estimator='henderson3', split_units=np.arange(rows.sum()))[2])
        print(f'fold {f} done', flush=True)
    scores = np.stack(scores)
    chosen = highest_level_within_one_se(RIDGES, scores.reshape((-1, len(RIDGES))))
    np.savez(args.out, date=args.date, folds=folds, ridges=RIDGES, chosen=chosen, scores=scores)
    print(f'date {miscanthus.DATES[args.date]}; chosen ridge {chosen:g}')
    print('held-out Henderson3 heritability of trait 1, mean over folds and splits:')
    for ridge, s in zip(RIDGES, scores.mean(axis=(0, 1)), strict=True):
        print(f'  {ridge:<9.3g} {s:.4f}')


if __name__ == '__main__':
    main()
