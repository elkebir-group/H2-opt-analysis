"""GWAS figures of the simulation (paper Fig. 3c-e and Fig. S2).

Writes into --out-dir: auc.pdf (AUC of causal-bin detection per trait), roc_trait{1,2,3}.pdf (mean
ROC curves) and gwas_{example-name}.png (Manhattan and QQ plots of the GWAS of one method's traits
1-3 on one instance).

    python pipelines/scripts/simulation/plot_gwas.py --roc roc.npz \\
        --gwas-example gwas/h2opt_rmsprop.npz --example-name h2opt_rmsprop_instance0 \\
        --example-label "H2-opt, instance 0" --out-dir figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis import simulation  # noqa: E402
from h2opt_analysis.figures import COLORS, save_trait_boxplots  # noqa: E402


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
    roc_methods = list(roc['methods'])
    values = {m: {t: roc['auc'][t, j] for t in range(n_t)} for j, m in enumerate(roc_methods)}
    save_trait_boxplots(out / 'auc.pdf', n_t, roc_methods, values, 'AUC of causal 100 kb bins')

    for t in range(n_t):
        fig, ax = plt.subplots(figsize=(4, 3.2))
        for j, m in enumerate(roc_methods):
            median = np.median(roc['auc'][t, j])
            ax.plot(roc['fpr'], roc['tpr'][t, j], color=COLORS[m],
                    label=f'{simulation.METHODS[m]}: AUC {median:.3f}')
        ax.plot([0, 1], [0, 1], 'k:', linewidth=1)
        ax.set_xlabel('false positive rate')
        ax.set_ylabel('true positive rate')
        ax.set_title(f'trait {t + 1}, mean over instances', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f'roc_trait{t + 1}.pdf')

    g = np.load(args.gwas_example)
    chromosome, position, score = g['chr'], g['ps'], g['neg_log10_p']
    offset = np.zeros(len(position), dtype=np.int64)
    start = 0
    for c in np.unique(chromosome):
        on = chromosome == c
        offset[on] = start
        start += position[on].max() + 1
    threshold = -np.log10(0.05 / score.shape[1])
    fig, axes = plt.subplots(n_t, 2, figsize=(8, 1.9 * n_t),
                             gridspec_kw={'width_ratios': [3, 1]})
    for t in range(n_t):
        ax = axes[t, 0]
        ax.scatter(position + offset, score[t], s=1, c=np.where(chromosome % 2, 'tab:red',
                                                                 'tab:blue'))
        ax.axhline(threshold, color='black', linestyle=':', linewidth=1)
        ax.set_ylabel(f'trait {t + 1}\n-log10 p')
        ax.set_xticks([])
        ax.spines[['top', 'right']].set_visible(False)
        ax = axes[t, 1]
        observed = np.sort(score[t])
        expected = -np.log10(np.arange(len(observed), 0, -1) / (len(observed) + 1))
        ax.scatter(expected, observed, s=1)
        top = min(expected.max(), observed.max())
        ax.plot([0, top], [0, top], 'r--', linewidth=1)
        ax.set_title(f'lambda GC {g["lambda_gc"][t]:.3f}', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1, 0].set_xlabel('genome position')
    axes[-1, 1].set_xlabel('expected -log10 p')
    fig.suptitle(args.example_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f'gwas_{args.example_name}.png', dpi=150)



if __name__ == '__main__':
    main()
