"""Permuted-genotype null of the whole H2-opt -> GWAS pipeline.

Each measured individual (row) is attributed to the genotype of another row, so no genotype-phenotype relation
remains while the genotype structure (group sizes per stratum) is kept. The permuted labels must be computed
once, on all rows, and used everywhere a genotype label is used: training H2-opt, the folds, heritability
evaluation and the per-genotype phenotypes given to GWAS (which then pairs them with the real SNPs of the
permuted label). Environment stays attached to the row. Subset rows (an environment, a fold) after permuting.
"""

import numpy as np


def permute_genotypes(genotypes, strata=None, seed=0):
    """Genotype labels permuted among the rows of each stratum (e.g. field set or darkness treatment).

    strata: (n,) or (n, k) array; rows with equal values form a stratum. None permutes across all rows."""
    genotypes = np.asarray(genotypes)
    if strata is None:
        groups = np.zeros(len(genotypes), dtype=int)
    else:
        strata = np.asarray(strata).reshape(len(genotypes), -1)
        _, groups = np.unique(strata, axis=0, return_inverse=True)
        groups = groups.reshape(-1)
    rng = np.random.default_rng(seed)
    permuted = genotypes.copy()
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        permuted[rows] = genotypes[rows[rng.permutation(len(rows))]]
    return permuted
