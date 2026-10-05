"""Shared plotting helpers: method colors and grouped boxplots."""

import matplotlib.pyplot as plt
import numpy as np

from .methods import LABELS

COLORS = {'truth': 'tab:red', 'h2opt_paper': 'tab:blue', 'h2opt': 'tab:blue',
          'pca': 'tab:orange', 'genetic_pca': 'tab:purple',
          'lda': 'tab:brown', 'pch': 'tab:pink', 'max_features': 'tab:green'}


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
    """Figure legend with one colored patch per method."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in slugs]
    fig.legend(handles, [LABELS[s] for s in slugs], loc='upper center',
               ncol=min(len(slugs), 4), frameon=False, fontsize=8)


def save_trait_boxplots(path, n_traits, slugs, values, ylabel, width=5):
    """Figure of grouped boxplots per trait (values[slug][trait] is a list over instances)."""
    fig, ax = plt.subplots(figsize=(width, 3.2))
    grouped_boxplot(ax, [str(t + 1) for t in range(n_traits)], slugs, values)
    ax.set_xlabel('trait')
    ax.set_ylabel(ylabel)
    legend(fig, slugs)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(path)
    plt.close(fig)


def plot_noise_choices(levels, chosen, scores, path):
    """Noise levels chosen by h2opt.selection.select_noise_levels in several subsets (folds).

    chosen: (folds, traits) levels; scores: (folds, traits, splits, levels). Left: the mean
    held-out heritability of trait 1 over the splits per level, one line per fold, its chosen
    level marked. Right: the chosen level of each trait, one dot per fold. Saves path (.pdf) and
    a .png next to it."""
    levels = list(levels)
    index = np.vectorize(levels.index)(chosen)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(levels))
    for fold, mean in enumerate(scores[:, 0].mean(axis=1)):
        left.plot(x, mean, '-', color='0.6', lw=1)
        left.plot(index[fold, 0], mean[index[fold, 0]], 'o', color='tab:blue', ms=4)
    left.set_xticks(x, [f'{lev:g}' for lev in levels])
    left.set_xlabel('noise SD / feature SD')
    left.set_ylabel('held-out heritability, trait 1')
    left.set_title('mean over splits; dot: chosen', fontsize=9)
    n_folds, n_traits = chosen.shape
    jitter = np.linspace(-0.25, 0.25, n_folds)
    for trait in range(n_traits):
        right.plot(trait + 1 + jitter, index[:, trait], 'o', color='tab:blue', ms=3)
    right.set_xticks(range(1, n_traits + 1))
    right.set_yticks(x, [f'{lev:g}' for lev in levels])
    right.set_xlabel('trait')
    right.set_ylabel('chosen noise SD / feature SD')
    right.set_title('one dot per fold', fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)
