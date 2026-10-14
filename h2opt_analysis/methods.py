"""Ways to get synthetic traits, by the slug used in file names, and the baselines' fits."""

from h2opt import baselines

# Figure label of each method. 'truth' are the simulated traits themselves.
LABELS = {
    'truth': 'ground truth',
    'h2opt': 'H2-opt',
    'h2opt_hybrid': 'H2-opt hybrid',
    'pca': 'PCA',
    'genetic_pca': 'genetic PCA',
    'pch': 'PCH',
    'max_features': 'single trait',
    'single_band': 'single band',
}

# Order of the methods in every figure: H2-opt first, then the baselines from the most to the
# least genetic information.
ORDER = ['h2opt_hybrid', 'h2opt', 'pch', 'genetic_pca', 'max_features', 'single_band', 'pca']


def ordered(slugs):
    """The slugs as a list, with the methods of ORDER in figure order.

    The methods move only among their own positions. Other entries (such as 'truth' or 'ndvi')
    keep their positions."""
    slugs = list(slugs)
    slots = [i for i, s in enumerate(slugs) if s in ORDER]
    for i, slug in zip(slots, sorted((slugs[i] for i in slots), key=ORDER.index), strict=True):
        slugs[i] = slug
    return slugs


BASELINES = {'pca': baselines.PCA, 'genetic_pca': baselines.GeneticPCA, 'pch': baselines.PCH,
             'max_features': baselines.MaxHeritabilityFeatures}


def fit_baseline(slug, X, groups, environment, rows, n_traits):
    """Traits of a baseline fitted on the selected rows, for all rows, and its tuning.

    PCH's ridge is chosen on the selected rows by PCH.tune, the selection procedure and rule of
    H2-opt's noise level (tuning: a dict of arrays ridge, ridges, ridge_scores,
    ridge_fold_scores). Other baselines have no tuning (empty dict).
    """
    tuning = {}
    if slug == 'pch':
        ridge, scores, fold_scores = baselines.PCH.tune(X[rows], groups[rows], environment[rows])
        baseline = baselines.PCH(n_traits, ridge=ridge)
        tuning = dict(ridge=ridge, ridges=baselines.RIDGES, ridge_scores=scores,
                      ridge_fold_scores=fold_scores)
    else:
        baseline = BASELINES[slug](n_traits)
    return baseline.fit(X[rows], groups[rows], environment[rows]).transform(X), tuning
