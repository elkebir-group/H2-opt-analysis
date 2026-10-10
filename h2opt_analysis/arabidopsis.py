"""Arabidopsis LC-MS metabolomics (Fernie lab): 1,112 samples of 280 accessions, two treatments.

Each sample is an LC-MS run binned by integer m/z and 30 s of retention time, for MS1 and MS/MS
(processed/imageMetab.npz: samples x 2 MS levels x 2,000 m/z x 60 bins, summed intensities). The
treatments are normal growth ('0d') and 6 days of darkness ('6d'), each analyzed on its own; every
accession has one sample per year in a treatment, and the year is the environment.

The nine labels 6909-1 ... 6909-9 are Col-0 (accession 6909 of the PLINK set): metabolite profiles
show no heritability among them, so they are pooled as one accession (POOLED).
"""

import numpy as np
from h2opt import load_npz
from h2opt.folds import per_group

from .paths import data

TREATMENTS = ('0d', '6d')
N_FOLDS = 5
PROCESSED = data('metab', 'metabData_Fernie', 'processed')
# m/z 0-1499 and retention times 0-20 min, the paper's crop.
N_MZ, N_RT = 1500, 40
# Labels of one accession: Col-0 sown as nine labeled lines.
POOLED = {f'6909-{k}': '6909' for k in range(1, 10)}
# PLINK set of the genotyped accessions (call method 75, TAIR8). The .fam beside it was overwritten
# with a subset by the original GWAS code; the full sample list of the .bed is the _copy.fam.
GENOTYPES = data('metab', 'metabData_Alex', 'call_method_75', 'call_method_75_TAIR8')
GENOTYPE_SAMPLES = GENOTYPES.with_name(GENOTYPES.name + '_copy.fam')


def precise_masses(treatment):
    """Path of the precise-mass sums of one treatment: (4, 2000 m/z bins, 60 retention-time bins,
    10001 steps) float64, the steps splitting each integer m/z bin into m/z - 0.5 + step / 10000;
    the 4 sums over the treatment's samples are of the intensity, the retention time (s) times the
    intensity, (unused), and the squared intensity."""
    return PROCESSED / f'massIntensitiesPrecise2_{treatment}.npz'


def labels():
    """Accession (pooled), treatment and year of every sample, in the order of imageMetab.npz."""
    names = load_npz(PROCESSED / 'names.npz').astype(str)
    accessions = np.array([POOLED.get(name, name) for name in names])
    return accessions, load_npz(PROCESSED / 'env.npz').astype(str), \
        load_npz(PROCESSED / 'year.npz').astype(str)


def features():
    """Measurements of all samples, (1112, 2 * N_MZ * N_RT): the cropped intensities over their
    largest value, flattened in the order (MS level, m/z, retention time) as in the paper."""
    X = load_npz(PROCESSED / 'imageMetab.npz')[:, :, :N_MZ, :N_RT]
    return (X / X.max()).reshape((len(X), -1))


def design(treatment):
    """Rows of one treatment among all samples, their accession labels and environment (year) as an
    (n, 1) array."""
    accessions, treatments, years = labels()
    rows = treatments == treatment
    return rows, accessions[rows], years[rows].reshape((-1, 1))


def load(features_path, treatment):
    """Measurements X of one treatment (from the .npy of all samples' features()), accession labels
    and environment (year) as an (n, 1) array."""
    rows, accessions, environment = design(treatment)
    return np.asarray(np.load(features_path, mmap_mode='r')[rows]), accessions, environment


def folds(accessions, seed=0):
    """Fold (0-4) of each sample; all samples of an accession share a fold."""
    return per_group(accessions, lambda c: np.random.RandomState(seed).randint(N_FOLDS, size=c))


def genotype_rows(accessions, fam):
    """Row in the .fam file fam of the accession of each sample, -1 for an accession without
    genotypes. The sample IDs of the .fam are accession numbers."""
    samples = np.loadtxt(fam, dtype=str, usecols=0, ndmin=1)
    row = {sample: r for r, sample in enumerate(samples)}
    return np.array([row.get(a, -1) for a in np.asarray(accessions).astype(str)])
