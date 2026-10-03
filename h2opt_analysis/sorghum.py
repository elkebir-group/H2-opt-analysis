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


def line_phenotypes(Y, genotypes, legacy=False):
    """Per-line phenotypes for GWAS, in .fam order: the mean of each standardized trait over the plants of the line.
    Lines without plants get NaN. legacy reproduces the paper's GWAS: each trait standardized over plants and clipped
    to [-2, 2] in its own precision (float32 for the paper's traits), means truncated to 6 characters, and lines
    without plants given 0."""
    Y = np.asarray(Y).reshape(len(Y), -1)
    if not legacy:
        Y = Y.astype(np.float64)
    names = line_names()
    plants = [np.flatnonzero(genotypes == name) for name in names]
    out = np.full((len(names), Y.shape[1]), 0.0 if legacy else np.nan)
    for t in range(Y.shape[1]):
        y = Y[:, t] - np.mean(Y[:, t])
        y = y / np.mean(y ** 2) ** 0.5
        if legacy:
            y = np.clip(y, -2, 2)
        for i, rows in enumerate(plants):
            if len(rows):
                out[i, t] = float(str(np.mean(y[rows]))[:6]) if legacy else np.mean(y[rows])
    return out
