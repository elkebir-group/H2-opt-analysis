"""Sorghum hyperspectral data: 1,920 plants x 2,151 wavelengths (350-2500 nm, 1 nm), genotypes and field sets."""

import numpy as np
from h2opt import load_npz

from .paths import data

N_FOLDS = 10


def load():
    """Measurements X, genotype labels and environment (field set) as an (n, 1) array."""
    folder = data('plant', 'processed', 'sor')
    X = load_npz(folder / 'X.npz')
    genotypes = load_npz(folder / 'names.npz')
    environment = load_npz(folder / 'set1.npz').reshape((-1, 1))
    return X, genotypes, environment


def folds(genotypes, seed=0):
    """Fold (0-9) of each plant; all plants of a genotype share a fold. The paper's folds use seed 0."""
    _, inverse = np.unique(genotypes, return_inverse=True)
    return np.random.RandomState(seed).randint(N_FOLDS, size=inverse.max() + 1)[inverse]
