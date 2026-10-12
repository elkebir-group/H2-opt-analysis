"""The figure order of the methods."""

from h2opt_analysis.methods import ordered


def test_ordered_saved_orders():
    saved = ['truth', 'h2opt', 'h2opt_hybrid', 'pca', 'genetic_pca', 'pch', 'max_features']
    assert ordered(saved) == ['truth', 'h2opt_hybrid', 'h2opt', 'pch', 'genetic_pca',
                              'max_features', 'pca']
    assert ordered(['h2opt_paper', 'pca', 'genetic_pca', 'pch', 'max_features']) == [
        'h2opt_paper', 'pch', 'genetic_pca', 'max_features', 'pca']


def test_ordered_variants_and_other_entries():
    assert ordered(['truth', 'h2opt_paper', 'pca', 'h2opt']) == [
        'truth', 'h2opt', 'h2opt_paper', 'pca']
    assert ordered(['h2opt_paper', 'h2opt', 'h2opt_hybrid', 'ndvi']) == [
        'h2opt_hybrid', 'h2opt', 'h2opt_paper', 'ndvi']
