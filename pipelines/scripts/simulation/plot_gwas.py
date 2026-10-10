"""GWAS figures of the simulation (paper Fig. 3c-e and Fig. S2).

Writes into --out-dir: auc.pdf (AUC of causal-bin detection per trait), roc_trait{1,2,3}.pdf (mean
ROC curves) and gwas_{example-name}.png (Manhattan and QQ plots of the GWAS of one method's traits
1-3 on one instance).

    python pipelines/scripts/simulation/plot_gwas.py --roc roc.npz \\
        --gwas-example gwas/h2opt.npz --example-name h2opt_instance0 \\
        --example-label "H2-opt, instance 0" --out-dir figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis import simulation  # noqa: E402
from h2opt_analysis.figures import COLORS, plot_gwas, save_trait_boxplots  # noqa: E402
from h2opt_analysis.methods import LABELS  # noqa: E402


def auc_values(roc):
    """AUC per method and trait of a roc.py result: the slugs and the values (values[slug][t] is
    an array over the instances)."""
    methods = list(roc['methods'])
    return methods, {m: {t: roc['auc'][t, j] for t in range(simulation.N_TRAITS)}
                     for j, m in enumerate(methods)}


def plot_roc(ax, roc, t):
    """Mean ROC curves of trait t (0-based) for each method of a roc.py result, with the median
    AUC over the instances in the legend."""
    for j, m in enumerate(roc['methods']):
        median = np.median(roc['auc'][t, j])
        ax.plot(roc['fpr'], roc['tpr'][t, j], color=COLORS[m],
                label=f'{LABELS[m]}: AUC {median:.3f}')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roc', required=True)
    parser.add_argument('--gwas-example', required=True)
    parser.add_argument('--example-name', required=True, help='file name stem for the example')
    parser.add_argument('--example-label', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_t = simulation.N_TRAITS

    roc = np.load(args.roc)
    slugs, values = auc_values(roc)
    save_trait_boxplots(out / 'auc.pdf', n_t, slugs, values, 'AUC of causal 100 kb bins')

    for t in range(n_t):
        fig, ax = plt.subplots(figsize=(4, 3.2))
        plot_roc(ax, roc, t)
        ax.set_title(f'trait {t + 1}, mean over instances', fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f'roc_trait{t + 1}.pdf')
        plt.close(fig)

    plot_gwas(out / f'gwas_{args.example_name}.png', np.load(args.gwas_example), range(n_t),
              [f'trait {t + 1}' for t in range(n_t)], args.example_label)


if __name__ == '__main__':
    main()
