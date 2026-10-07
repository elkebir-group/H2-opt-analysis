"""Choose the noise level of Miscanthus H2-opt (the image CNN) in each fold, one for all dates.

From the scores of every level (score_noise.py, one file per level), the level with the best
held-out heritability averaged over the dates and splits (h2opt.selection.best_level, the rule of
select_noise_level). Writes an .npz (levels; chosen: one level per fold; noise_sd: folds x dates,
the standard deviation of the chosen noise on each date's images; scores: folds x dates x 1 x
splits x levels) read by train_h2opt.py, and a figure of the choices.

    python pipelines/scripts/miscanthus/choose_noise.py --scores level0.1.npz level1.0.npz \\
        --out noise.npz --figure noise.pdf
"""

import argparse

import numpy as np
from h2opt.selection import best_level

from h2opt_analysis import miscanthus
from h2opt_analysis.figures import plot_noise_choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', nargs='+', required=True, help='score_noise.py outputs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--figure', required=True)
    args = parser.parse_args()

    runs = sorted((np.load(path) for path in args.scores), key=lambda run: float(run['level']))
    levels = np.array([float(run['level']) for run in runs])
    scores = np.stack([run['scores'] for run in runs], axis=-1)
    scale = runs[0]['scale']
    folds = np.arange(len(scores))
    chosen = np.array([best_level(levels, score) for score in scores])
    np.savez(args.out, folds=folds, dates=miscanthus.DATES, levels=levels, chosen=chosen,
             noise_sd=chosen[:, None] * scale, scores=scores)
    print('chosen noise level; held-out heritability per level (mean over dates and splits)')
    print('  levels:', ' '.join(f'{lev:.3g}' for lev in levels))
    for fold, level, score in zip(folds, chosen, scores, strict=True):
        means = score.mean(axis=(0, 1, 2))
        print(f'  fold {fold}: {level:.3g};', ' '.join(f'{m:.3f}' for m in means))
    plot_noise_choices(levels, chosen, scores[:, :, 0], args.figure, series='date',
                       series_labels=miscanthus.DATES)


if __name__ == '__main__':
    main()
