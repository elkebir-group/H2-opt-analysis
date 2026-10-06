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


def plot_noise_choices(levels, chosen, scores, path, series='trait', series_labels=None):
    """Noise levels chosen by h2opt.selection.select_noise_level in several subsets (folds).

    chosen: (folds,) levels; scores: (folds, series, splits, levels), the series being traits or
    measurement sets (e.g. dates; series names them, series_labels gives their labels, default
    1, 2, ...). Left: the held-out heritability averaged over the series and splits per level, one
    line per fold, its chosen level marked. Right: the held-out heritability of each series
    averaged over the folds and splits per level. Saves path (.pdf) and a .png next to it."""
    levels = np.asarray(levels)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for fold, mean in enumerate(scores.mean(axis=(1, 2))):
        left.plot(levels, mean, '-', color='0.6', lw=1)
        a = int(np.argmin(np.abs(levels - chosen[fold])))
        left.plot(levels[a], mean[a], 'o', color='tab:blue', ms=4)
    left.set_ylabel(f'held-out heritability, mean of {series}s')
    left.set_title('one line per fold; dot: chosen', fontsize=9)
    n_series = scores.shape[1]
    labels = series_labels if series_labels is not None else range(1, n_series + 1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_series))
    for t, label in zip(range(n_series), labels, strict=True):
        right.plot(levels, scores[:, t].mean(axis=(0, 1)), '-', color=colors[t], lw=1,
                   label=f'{label}')
    right.set_ylabel('held-out heritability')
    right.set_title('mean over folds', fontsize=9)
    right.legend(title=series, frameon=False, fontsize=6, title_fontsize=7,
                 ncol=2 if n_series > 12 else 1, loc='center left', bbox_to_anchor=(1, 0.5))
    for ax in (left, right):
        ax.set_xscale('log')
        ax.set_xlabel('noise SD / feature SD')
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)
