"""Figures of the simulation (paper Fig. 3 and Fig. S2).

Writes into --out-dir: heritability.pdf (test heritability of synthetic traits 1-5 and of the true
traits), correlation.pdf (|r| of true trait t with synthetic trait t), auc.pdf (AUC of causal-bin
detection per trait), roc_trait{1,2,3}.pdf (mean ROC curves) and gwas_instance{k}.png (Manhattan
and QQ plots of the GWAS of one method's traits 1-3 on one instance).

    python pipelines/scripts/simulation/plot.py --evaluation evaluation.npz --roc roc.npz \\
        --gwas-example gwas_h2opt_rmsprop.npz --example-label "H2-opt, instance 0" \\
        --out-dir figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis import simulation  # noqa: E402

COLORS = {'truth': 'tab:red', 'h2opt_rmsprop': 'tab:blue', 'h2opt_lbfgs': 'tab:cyan',
          'pca': 'tab:orange', 'genetic_pca': 'tab:purple', 'lda': 'tab:brown', 'pch': 'tab:pink',
          'max_features': 'tab:green'}


def grouped_boxplot(ax, groups, slugs, values):
    """Boxplots of values[slug][group] (a list per group), side by side within each group."""
    width = 0.8 / len(slugs)
    for s, slug in enumerate(slugs):
        positions = [g + (s - (len(slugs) - 1) / 2) * width for g in range(len(groups))]
        data = [values[slug].get(g, []) for g in range(len(groups))]
        keep = [k for k, d in enumerate(data) if len(d)]
        boxes = ax.boxplot([data[k] for k in keep], positions=[positions[k] for k in keep],
                           widths=width * 0.9, patch_artist=True, showfliers=False,
                           medianprops={'color': 'black'})
        for box in boxes['boxes']:
            box.set(facecolor=COLORS[slug], edgecolor=COLORS[slug], alpha=0.8)
    ax.set_xticks(range(len(groups)), groups)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.9')
    ax.set_axisbelow(True)


def legend(fig, slugs):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in slugs]
    fig.legend(handles, [simulation.METHODS[s] for s in slugs], loc='upper center',
               ncol=min(len(slugs), 4), frameon=False, fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--roc', required=True)
    parser.add_argument('--gwas-example', required=True)
    parser.add_argument('--example-name', required=True, help='file name stem for the example')
    parser.add_argument('--example-label', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_t = simulation.N_TRAITS

    ev = np.load(args.evaluation)
    methods = list(ev['methods'])
    herit = np.clip(ev['herit'][:, :, 1], 0, None)  # methods x instances x traits, test plants
    k = herit.shape[2]
    slugs = ['truth'] + methods
    values = {'truth': {t: np.clip(ev['herit_truth'][:, 1, t], 0, None) for t in range(n_t)}}
    values.update({m: {t: herit[j, :, t] for t in range(k)} for j, m in enumerate(methods)})
    fig, ax = plt.subplots(figsize=(7, 3.2))
    grouped_boxplot(ax, [str(t + 1) for t in range(k)], slugs, values)
    ax.set_xlabel('trait')
    ax.set_ylabel('test heritability')
    legend(fig, slugs)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out / 'heritability.pdf')

    diagonal = np.abs(ev['cor'][:, :, np.arange(n_t), np.arange(n_t)])
    values = {m: {t: diagonal[j, :, t] for t in range(n_t)} for j, m in enumerate(methods)}
    fig, ax = plt.subplots(figsize=(5, 3.2))
    grouped_boxplot(ax, [str(t + 1) for t in range(n_t)], methods, values)
    ax.set_xlabel('trait')
    ax.set_ylabel('|r| with true trait')
    legend(fig, methods)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out / 'correlation.pdf')

    roc = np.load(args.roc)
    roc_methods = list(roc['methods'])
    values = {m: {t: roc['auc'][t, j] for t in range(n_t)} for j, m in enumerate(roc_methods)}
    fig, ax = plt.subplots(figsize=(5, 3.2))
    grouped_boxplot(ax, [str(t + 1) for t in range(n_t)], roc_methods, values)
    ax.set_xlabel('trait')
    ax.set_ylabel('AUC of causal 100 kb bins')
    legend(fig, roc_methods)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out / 'auc.pdf')

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
