"""Ways to get synthetic traits, by the slug used in file names, and the baselines' fits."""

from h2opt import baselines

# Figure label of each method. 'truth' are the simulated traits themselves; 'h2opt_paper' the
# predictions behind the paper.
LABELS = {
    'truth': 'ground truth',
    'h2opt_paper': 'H2-opt (paper)',
    'h2opt': 'H2-opt',
    'pca': 'PCA',
    'genetic_pca': 'genetic PCA',
    'lda': 'LDA',
    'pch': 'PCH',
    'max_features': 'single trait',
}

BASELINES = {'pca': baselines.PCA, 'genetic_pca': baselines.GeneticPCA, 'lda': baselines.LDA,
             'pch': baselines.PCH, 'max_features': baselines.MaxHeritabilityFeatures}


def fit_baseline(slug, X, groups, environment, rows, n_traits):
    """Traits of a baseline fitted on the selected rows, for all rows.

    PCH's ridge is chosen on the selected rows by cross-validated heritability (PCH.tune).
    """
    if slug == 'pch':
        ridge, _, _ = baselines.PCH.tune(X[rows], groups[rows], environment[rows], n_traits)
        baseline = baselines.PCH(n_traits, ridge=ridge)
    else:
        baseline = BASELINES[slug](n_traits)
    return baseline.fit(X[rows], groups[rows], environment[rows]).transform(X)
