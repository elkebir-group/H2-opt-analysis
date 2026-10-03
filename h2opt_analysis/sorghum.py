"""Sorghum hyperspectral data: 1,920 plants x 2,151 wavelengths (350-2500 nm, 1 nm).

Also the genotype labels, field sets, folds and the PLINK set of the sequenced lines.
"""

import numpy as np
from h2opt import load_npz

from .paths import data

N_FOLDS = 10
FIRST_WAVELENGTH = 350  # nm; column j of X is wavelength 350 + j nm


def band(X, low, high):
    """Mean measurement over wavelengths low to high nm (inclusive)."""
    return X[:, low - FIRST_WAVELENGTH:high - FIRST_WAVELENGTH + 1].mean(axis=1)


def ndvi(X):
    """NDVI with 20 nm bands centered at 680 nm (red) and 800 nm (NIR) (Sims and Gamon, 2002)."""
    red, nir = band(X, 670, 689), band(X, 790, 809)
    return (nir - red) / (nir + red)


def load():
    """Measurements X, genotype labels and environment (field set) as an (n, 1) array."""
    folder = data('plant', 'processed', 'sor')
    X = load_npz(folder / 'X.npz')
    genotypes = load_npz(folder / 'names.npz')
    environment = load_npz(folder / 'set1.npz').reshape((-1, 1))
    return X, genotypes, environment


def folds(genotypes, seed=0):
    """Fold (0-9) of each plant; all plants of a genotype share a fold. The paper used seed 0."""
    _, inverse = np.unique(genotypes, return_inverse=True)
    return np.random.RandomState(seed).randint(N_FOLDS, size=inverse.max() + 1)[inverse]


def paper_traits(fold):
    """Path of the paper's linear H2-opt predictions for a genotype fold (10 traits, all plants)."""
    return data('plant', 'syntheticTraits', f'linear_crossVal_reg4_{fold}_mod10.npz')


def load_traits(path):
    """Traits of all plants from an .npz: the pipeline's 'traits' array, or the paper's arr_0."""
    with np.load(path) as f:
        return f['traits'] if 'traits' in f else f['arr_0']


GENOTYPES = data('software', 'data', 'WEST_original')  # PLINK set of the 836 sequenced lines


def line_accessions():
    """Accessions (genotype labels of the plants) of each line of the .fam file, in .fam order.

    From nameKey_corrected.tsv, which follows Supplemental Table S1 of Ferguson et al. (2021): 8
    sequenced lines stand for several field accessions each (2-7, of the same landrace), all
    assigned the line's genotype."""
    key = np.loadtxt(data('software', 'data', 'nameKey_corrected.tsv'), dtype=str, delimiter='\t',
                     skiprows=1)
    accessions = {}
    for fam_id, _, name in key:
        accessions.setdefault(fam_id, []).append(name.replace(' ', ''))
    return [accessions[f] for f in np.loadtxt(f'{GENOTYPES}.fam', dtype=str, usecols=0)]


def line_phenotypes(Y, genotypes):
    """Per-line phenotypes for GWAS, in .fam order.

    The phenotype of a line is the mean of each standardized trait over the plants of all
    accessions of the line. Lines without plants get NaN.
    """
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    lines = line_accessions()
    out = np.full((len(lines), Y.shape[1]), np.nan)
    for i, accessions in enumerate(lines):
        plants = np.isin(genotypes, accessions)
        if plants.any():
            out[i] = Y[plants].mean(axis=0)
    return out
