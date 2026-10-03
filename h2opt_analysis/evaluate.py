"""Train/test heritability of synthetic traits and baselines."""

import torch
from h2opt import anova_heritability, baselines


def heritability(Y, groups, environment, rows):
    """ANOVA heritability of each column of the array Y on the selected rows."""
    traits = torch.tensor(Y[rows]).float()
    return anova_heritability(traits, groups[rows], environment[rows]).numpy()


def fit_baseline(name, X, groups, environment, rows, n_traits):
    """Traits of a baseline fitted on the selected rows, for all rows.

    PCH's ridge is chosen on the selected rows by cross-validated heritability (PCH.tune).
    """
    if name == 'PCH':
        ridge, _, _ = baselines.PCH.tune(X[rows], groups[rows], environment[rows], n_traits)
        baseline = baselines.PCH(n_traits, ridge=ridge)
    else:
        baseline = {'PCA': baselines.PCA, 'genetic PCA': baselines.GeneticPCA, 'LDA': baselines.LDA,
                    'max features': baselines.MaxHeritabilityFeatures}[name](n_traits)
    return baseline.fit(X[rows], groups[rows], environment[rows]).transform(X)
