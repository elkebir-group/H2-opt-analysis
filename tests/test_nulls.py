import numpy as np

from h2opt_analysis import nulls
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


def test_save_keeps_arrays_scores_and_summary(tmp_path):
    score = np.zeros((2, 100), dtype=np.float32)
    score[1, :2] = 4.0
    scores = {'chr': np.ones(100, dtype=np.int16), 'ps': np.arange(100), 'neg_log10_p': score,
              'lambda_gc': np.array([1.0, 0.9])}
    nulls.save(tmp_path / 'perm0.npz', scores, [0.3, 0.2], [0.0, 0.1], permutation=0,
               labels=np.array(['a', 'b']))
    with np.load(tmp_path / 'perm0.npz') as f:
        assert f['hits'].tolist() == [0, 2] and f['labels'].tolist() == ['a', 'b']
        assert np.array_equal(f['neg_log10_p'], score) and f['h2_test'].tolist() == [0.0, 0.1]
