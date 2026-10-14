"""Sorghum hyperspectral data: 1,920 plants x 2,151 wavelengths (350-2500 nm, 1 nm).

Also the genotype labels, field sets, folds and the PLINK set of the sequenced lines.
"""

import csv
import re

import numpy as np
from h2opt import load_npz
from h2opt.folds import per_group

from .paths import data

N_FOLDS = 10
FIRST_WAVELENGTH = 350  # nm; column j of X is wavelength 350 + j nm
PROCESSED = data('plant', 'processed', 'sor')
SPECTRA = PROCESSED / 'X.npz'  # plants x wavelengths
GENOTYPES = data('software', 'data', 'WEST_original')  # PLINK set of the 836 sequenced lines
# Genotype means of the measured traits (Ferguson et al. 2021, Supplemental Table S12).
MEASURED_TRAITS = data('plant', 'processed', 'traits', 'SorgumTraits.tsv')


def band(X, low, high):
    """Mean measurement over wavelengths low to high nm (inclusive)."""
    return X[:, low - FIRST_WAVELENGTH:high - FIRST_WAVELENGTH + 1].mean(axis=1)


def ndvi(X):
    """NDVI with 20 nm bands centered at 680 nm (red) and 800 nm (NIR) (Sims and Gamon, 2002)."""
    red, nir = band(X, 670, 689), band(X, 790, 809)
    return (nir - red) / (nir + red)


def load():
    """Measurements X, genotype labels and environment (field set) as an (n, 1) array."""
    X = load_npz(SPECTRA)
    genotypes = load_npz(PROCESSED / 'names.npz')
    environment = load_npz(PROCESSED / 'set1.npz').reshape((-1, 1))
    return X, genotypes, environment


def folds(genotypes, seed=0):
    """Fold (0-9) of each plant; all plants of a genotype share a fold. The paper used seed 0."""
    return per_group(genotypes, lambda c: np.random.RandomState(seed).randint(N_FOLDS, size=c))


def load_traits(path):
    """Traits of all plants from an .npz (array 'traits')."""
    with np.load(path) as f:
        return f['traits']


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


def line_of_plant(genotypes):
    """Row in the .fam of GENOTYPES of each plant's line (see line_accessions), -1 for a plant of
    no sequenced line."""
    row = {}
    for i, accessions in enumerate(line_accessions()):
        for accession in accessions:
            if accession in row:
                raise ValueError(f'accession {accession} is in lines {row[accession]} and {i}')
            row[accession] = i
    return np.array([row.get(g, -1) for g in genotypes])


def measured_traits():
    """Trait names (column names of MEASURED_TRAITS), genotype labels and (genotypes, traits)
    values of the measured traits, NaN where empty."""
    with open(MEASURED_TRAITS, newline='') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    header = next(r for r in rows if r and r[0] == 'Genotype')
    n = header.index('', 1)  # the trait columns end at the first empty column
    body = [r for r in rows[rows.index(header) + 1:] if r and r[0]]
    values = np.array([[float(v) if v else np.nan for v in r[1:n]] for r in body])
    return header[1:n], np.array([r[0] for r in body]), values


def trait_slug(name):
    """File name form of a measured trait name: 'ci/ca (2017 model)' -> 'ci_ca_2017_model'."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
