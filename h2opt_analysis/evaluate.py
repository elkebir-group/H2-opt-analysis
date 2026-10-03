"""Train/test heritability of synthetic traits and baselines."""

import numpy as np
import torch
from h2opt import anova_heritability, baselines

RIDGES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def heritability(Y, groups, environment, rows):
    """ANOVA heritability of each column of the array Y on the selected rows."""
    traits = torch.tensor(Y[rows]).float()
    return anova_heritability(traits, groups[rows], environment[rows]).numpy()


def validation_rows(groups, rows, seed=123, n_parts=5):
    """Split rows by group into inner training and validation rows (1 in n_parts validates)."""
    _, inverse = np.unique(groups, return_inverse=True)
    is_val = np.random.RandomState(seed).randint(n_parts, size=inverse.max() + 1)[inverse] == 0
    return rows & ~is_val, rows & is_val


def tune_pch_ridge(X, groups, environment, rows, n_traits, ridges=RIDGES):
    """Ridge that maximizes mean validation heritability of PCH fitted on inner training rows."""
    fit, val = validation_rows(groups, rows)
    scores = []
    for ridge in ridges:
        pch = baselines.PCH(n_traits, ridge=ridge).fit(X[fit], groups[fit], environment[fit])
        Y = pch.transform(X)
        scores.append(heritability(Y, groups, environment, val).mean())
    return ridges[int(np.argmax(scores))]


def fit_baseline(name, X, groups, environment, rows, n_traits):
    """Traits of a baseline fitted on the selected rows, for all rows."""
    if name == 'PCH':
        ridge = tune_pch_ridge(X, groups, environment, rows, n_traits)
        baseline = baselines.PCH(n_traits, ridge=ridge)
    else:
        baseline = {'PCA': baselines.PCA, 'genetic PCA': baselines.GeneticPCA, 'LDA': baselines.LDA,
                    'max features': baselines.MaxHeritabilityFeatures}[name](n_traits)
    return baseline.fit(X[rows], groups[rows], environment[rows]).transform(X)
