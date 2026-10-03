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


GENOTYPES = data('software', 'data', 'WEST_original')  # PLINK set of the 836 sequenced lines


def line_names():
    """Name, as in the genotype labels, of each line of the genotype .fam file (the first synonym in the name key)."""
    key = np.loadtxt(data('software', 'data', 'nameKey_corrected.tsv'), dtype=str, delimiter='\t', skiprows=1)
    first = {}
    for fam_id, _, name in key:
        first.setdefault(fam_id, name.replace(' ', ''))
    return np.array([first[f] for f in np.loadtxt(f'{GENOTYPES}.fam', dtype=str, usecols=0)])


def line_phenotypes(Y, genotypes):
    """Per-line phenotypes for GWAS, in .fam order: the mean of each standardized trait over the plants of the line.
    Lines without plants get NaN."""
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    names = line_names()
    out = np.full((len(names), Y.shape[1]), np.nan)
    for i, name in enumerate(names):
        plants = genotypes == name
        if plants.any():
            out[i] = Y[plants].mean(axis=0)
    return out
