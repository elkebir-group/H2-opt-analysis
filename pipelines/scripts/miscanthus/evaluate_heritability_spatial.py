"""Henderson3 test heritability of the Miscanthus traits after a nearest-neighbor field
adjustment (Papadakis 1937; h2opt_analysis.miscanthus.papadakis), every method, date and fold.

The traits are read as in evaluate_heritability.py, for the dates --dates (indices into
miscanthus.DATES; one --by-date file per listed date, --by-fold files hold all dates). For each
fold, trait 1 of all kept plants is adjusted by the mean of the plants of the other-family plots
within --width field columns in the
same rep (the slope fitted on the fold's test plants) and scored by Henderson3 (families, plots
within families, rep and field column fixed) on the test plants. Central only: the measured
traits (measured_correlations.py) are scored on all plants with a value, before and after the
same adjustment, as a control. Writes an .npz (methods; dates; test: methods x dates x folds;
for Central measured_traits, measured_h2 and measured_h2_adjusted) and prints the mean
narrow-sense test heritability (4 x Henderson3, clipped to [0, 1] per fold) per date.

    python pipelines/scripts/miscanthus/evaluate_heritability_spatial.py --population central \\
        --by-date h2opt_hybrid date0.npz ... date13.npz --by-fold pch fold0.npz ... fold4.npz \\
        --dates 0 1 ... 13 --width 3 --out heritability_spatial.npz
"""

import argparse

import numpy as np
from h2opt import heritability
from measured_correlations import MEASURED_TRAITS, measured_traits

from h2opt_analysis import miscanthus


def h3(y, design, rows):
    return float(np.ravel(heritability(y[rows, None], design.families[rows],
                                       design.environment[rows], design.plots[rows],
                                       'henderson3'))[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--by-date', nargs='+', action='append', default=[],
                        metavar=('METHOD', 'FILE'), help='a method and one file per date')
    parser.add_argument('--by-fold', nargs='+', action='append', default=[],
                        metavar=('METHOD', 'FILE'), help='a method and one file per fold')
    parser.add_argument('--dates', type=int, nargs='+', required=True,
                        help='indices into miscanthus.DATES')
    parser.add_argument('--width', type=int, required=True, help='neighbor columns on each side')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    weights = miscanthus.neighbor_weights(design, args.width)
    dates, n_folds = args.dates, miscanthus.N_FOLDS
    n_dates = len(dates)
    traits = {}  # method -> dates x folds x plants (trait 1)
    for method, *files in args.by_date:
        assert len(files) == n_dates, f'{method}: give one file per date'
        traits[method] = np.stack([np.load(f)['traits'][..., 0] for f in files])
    for method, *files in args.by_fold:
        assert len(files) == n_folds, f'{method}: give one file per fold'
        traits[method] = np.stack([np.load(f)['traits'][dates, ..., 0] for f in files], axis=1)
    methods = list(traits)
    test = np.zeros((len(methods), n_dates, n_folds))
    for m, method in enumerate(methods):
        for d in range(n_dates):
            for f in range(n_folds):
                rows = design.fold == f
                test[m, d, f] = h3(miscanthus.papadakis(traits[method][d, f], weights, rows),
                                   design, rows)
        print(f'{method} done', flush=True)
    out = dict(methods=methods, dates=[miscanthus.DATES[d] for d in dates], test=test,
               width=args.width)

    if args.population == 'central':
        names = list(MEASURED_TRAITS['central'])
        values = measured_traits('central', design.keep)
        every = np.ones(len(design.families), bool)
        before, after = [], []
        for y in values:
            have = np.isfinite(y)
            before.append(h3(y, design, have))
            after.append(h3(miscanthus.papadakis(y, weights, every), design, have))
        out.update(measured_traits=names, measured_h2=before, measured_h2_adjusted=after)
        print('measured traits, narrow-sense heritability (4 x Henderson3), all plants: '
              'before -> after the adjustment')
        for name, b, a in zip(names, before, after, strict=True):
            print(f'  {name:28s} {4 * b:6.2f} -> {4 * a:6.2f}')
    np.savez(args.out, **out)

    print(f'mean narrow-sense test heritability of trait 1 after the adjustment (width '
          f'{args.width}; 4 x Henderson3, clipped to [0, 1] per fold), per date')
    print(f'{"date":8s}' + ''.join(f'{m:>14s}' for m in methods))
    for d, date in enumerate(out['dates']):
        row = np.clip(4 * test[:, d], 0, 1).mean(axis=1)
        print(f'{date:8s}' + ''.join(f'{v:14.3f}' for v in row))


if __name__ == '__main__':
    main()
