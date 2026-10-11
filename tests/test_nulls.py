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


def test_permute_units_moves_whole_units_within_strata():
    from h2opt_analysis.nulls import permute_units
    units = np.repeat(np.arange(12), 3)          # 12 plots of 3 plants
    strata = np.repeat(np.arange(3), 12)         # 3 reps of 4 plots
    family = np.repeat(np.arange(12) % 5, 3)     # one family per plot
    source = permute_units(units, strata, seed=1)
    assert sorted(source) == list(range(36))
    assert (strata[source] == strata).all()
    for u in range(12):
        rows = np.flatnonzero(units == u)
        # a unit takes all rows of one source unit, in order, and so one family
        assert len(set(units[source[rows]])) == 1
        assert np.all(np.diff(source[rows]) == 1)
        assert len(set(family[source[rows]])) == 1
    assert not np.array_equal(source, np.arange(36))
    assert np.array_equal(source, permute_units(units, strata, seed=1))
