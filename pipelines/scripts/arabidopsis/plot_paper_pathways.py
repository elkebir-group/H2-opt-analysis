"""Pathway enrichment of the H2-opt traits on Arabidopsis (images/pathwayAll and panel e of
images/metabFig).

A heatmap per treatment: the p-value (P(Gamma), MetaboAnalyst's mummichog p-value) of each pathway
for each trait, on a log color scale; a dot marks p < 0.05. pathwayAll shows every pathway with
p < 0.05 for a trait of either treatment; panel e shows the TOP pathways with the lowest p-value
over all traits and treatments. Reads the tables of pathway_enrichment.R, traits 1 to k per
treatment. Saves each .pdf and a .png next to it.

    python pipelines/scripts/arabidopsis/plot_paper_pathways.py --treatments 0d 6d \\
        --tables 0d/enrichment/trait{1..10}_pathways.csv 6d/enrichment/trait{1..10}_pathways.csv \\
        --out-all pathwayAll.pdf --out-top metabFig_e.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

P_COLUMN = 'P(Gamma)'
ALPHA = 0.05
TOP = 5


def load(tables, n_treatments):
    """p-values (treatments x traits x pathways, NaN for a pathway that was not tested) and the
    pathway names."""
    per_treatment = np.array_split(np.array(tables), n_treatments)
    frames = [[pd.read_csv(path, index_col=0)[P_COLUMN] for path in paths]
              for paths in per_treatment]
    pathways = sorted(set().union(*(f.index for row in frames for f in row)))
    p = np.array([[f.reindex(pathways).to_numpy() for f in row] for row in frames])
    return p, pathways


def heatmap(p, pathways, treatments, path):
    """One panel per treatment: pathways as rows, traits as columns."""
    n_traits = p.shape[1]
    fig, axes = plt.subplots(1, len(treatments), sharey=True, squeeze=False,
                             figsize=(2.0 + 2.2 * len(treatments), 0.9 + 0.22 * len(pathways)))
    norm = LogNorm(vmin=1e-3, vmax=1)
    for ax, treatment, values in zip(axes[0], treatments, p, strict=True):
        ax.imshow(np.clip(values.T, 1e-3, 1), cmap='magma', norm=norm, aspect='auto')
        rows, cols = np.nonzero(values.T < ALPHA)
        ax.scatter(cols, rows, s=4, c='white')
        ax.set_title(f'{treatment.removesuffix("d")} day')
        ax.set_xticks(range(n_traits), [str(t + 1) for t in range(n_traits)], fontsize=6)
        ax.set_xlabel('synthetic trait', fontsize=7)
    axes[0, 0].set_yticks(range(len(pathways)), pathways, fontsize=6)
    fig.colorbar(axes[0, 0].images[0], ax=axes[0].tolist(), shrink=0.8)
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatments', nargs='+', required=True)
    parser.add_argument('--tables', nargs='+', required=True,
                        help='pathway tables, traits 1 to k of each treatment in order')
    parser.add_argument('--out-all', required=True)
    parser.add_argument('--out-top', required=True)
    args = parser.parse_args()
    if len(args.tables) % len(args.treatments):
        parser.error('give the same number of tables for each treatment')

    p, pathways = load(args.tables, len(args.treatments))
    best = np.nanmin(p, axis=(0, 1))
    significant = [i for i in range(len(pathways)) if best[i] < ALPHA]
    top = sorted(np.argsort(best)[:TOP])
    for name, keep, out in (('all', significant, args.out_all), ('top', top, args.out_top)):
        heatmap(p[:, :, keep], [pathways[i] for i in keep], args.treatments, out)
        print(f'{name}: {len(keep)} pathways: ' + '; '.join(
            f'{pathways[i]} (min p {best[i]:.2g})' for i in keep), flush=True)


if __name__ == '__main__':
    main()
