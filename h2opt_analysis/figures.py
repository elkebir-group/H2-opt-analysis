"""Shared plotting helpers: method colors and grouped boxplots."""

import matplotlib.pyplot as plt

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


def plot_noise_scores(levels, scores, level, path):
    """Held-out heritability of trait 1 per noise level (h2opt.selection.select_noise_level):
    one line per validation split and their mean, the chosen level marked. Saves path (.pdf) and
    a .png next to it."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x = range(len(levels))
    for split, row in enumerate(scores):
        ax.plot(x, row, 'o-', color='0.6', lw=1, ms=3, label='split' if split == 0 else None)
    ax.plot(x, scores.mean(axis=0), 'o-', color='tab:blue', label='mean')
    ax.axvline(list(levels).index(level), color='k', ls='--', lw=1, label='chosen')
    ax.set_xticks(x, [f'{lev:g}' for lev in levels])
    ax.set_xlabel('noise SD / feature SD')
    ax.set_ylabel('held-out heritability, trait 1')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)
