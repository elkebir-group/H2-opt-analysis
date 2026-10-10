"""Train and test heritability of every method's traits on Arabidopsis (images/ArabiHeritTrain).

One row per treatment, train heritability left and test heritability right: boxplots over the
folds per trait, H2-opt and every baseline side by side. Reads the .npz of
evaluate_heritability.py, one per treatment. Saves --out (.pdf) and a .png next to it.

    python pipelines/scripts/arabidopsis/plot_paper_heritability.py --treatments 0d 6d \\
        --heritability 0d/heritability.npz 6d/heritability.npz --out ArabiHeritTrain.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis.figures import grouped_boxplot, legend  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatments', nargs='+', required=True)
    parser.add_argument('--heritability', nargs='+', required=True, help='one per treatment')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if len(args.treatments) != len(args.heritability):
        parser.error('give one heritability file per treatment')

    fig, axes = plt.subplots(len(args.treatments), 2, figsize=(11, 2.8 * len(args.treatments)),
                             sharey=True, squeeze=False)
    for row, (treatment, path) in enumerate(zip(args.treatments, args.heritability,
                                                strict=True)):
        results = np.load(path)
        methods = list(results['methods'])
        for ax, split in zip(axes[row], ('train', 'test'), strict=True):
            values = results[split]  # methods x folds x traits
            boxes = {m: {t: values[i, :, t] for t in range(values.shape[2])}
                     for i, m in enumerate(methods)}
            grouped_boxplot(ax, [str(t + 1) for t in range(values.shape[2])], methods, boxes)
            ax.set_title(f'{treatment.removesuffix("d")} day, {split}', fontsize=10)
            ax.set_xlabel('trait')
            ax.set_ylim(-0.02, 1.02)
            print(treatment, split, 'mean over folds per trait:')
            for i, m in enumerate(methods):
                print(f'  {m:13s}', np.round(values[i].mean(axis=0), 3).tolist(),
                      f'mean {values[i].mean():.3f}')
        axes[row, 0].set_ylabel('heritability (ANOVA)')
    legend(fig, methods)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out)
    fig.savefig(args.out.removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
