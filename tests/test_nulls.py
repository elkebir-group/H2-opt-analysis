import numpy as np

from h2opt_analysis.nulls import permute_genotypes


def test_permutation_within_strata():
    rng = np.random.default_rng(0)
    genotypes = np.repeat(np.arange(50).astype(str), 2)
    # genotypes nested in strata
    strata = np.column_stack([np.repeat(np.arange(5), 20), np.zeros(100)])
    permuted = permute_genotypes(genotypes, strata, seed=3)
    assert (permuted != genotypes).mean() > 0.5
    for s in range(5):
        rows = strata[:, 0] == s
        # same labels and group sizes per stratum
        assert sorted(permuted[rows]) == sorted(genotypes[rows])
    assert np.array_equal(permuted, permute_genotypes(genotypes, strata, seed=3))
    assert not np.array_equal(permuted, permute_genotypes(genotypes, strata, seed=4))
    assert sorted(permute_genotypes(genotypes, seed=rng.integers(100))) == sorted(genotypes)
