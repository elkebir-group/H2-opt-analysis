from pathlib import Path

import numpy as np
import pytest

from h2opt_analysis import gwas


def write_plink(prefix, G):
    """PLINK set from genotypes G (individuals x SNPs, counts of allele 1, NaN = missing)."""
    n, m = G.shape
    code = np.where(np.isnan(G), 1, np.select([G == 2, G == 1], [0, 2], 3)).T.astype(np.uint8)
    code = np.pad(code, ((0, 0), (0, (-n) % 4))).reshape(m, -1, 4)
    with open(f'{prefix}.bed', 'wb') as f:
        packed = code[..., 0] | code[..., 1] << 2 | code[..., 2] << 4 | code[..., 3] << 6
        f.write(b'\x6c\x1b\x01' + packed.tobytes())
    bim = [[1, f'snp{j}', 0, j + 1, 'A', 'C'] for j in range(m)]
    np.savetxt(f'{prefix}.bim', bim, fmt='%s', delimiter='\t')
    np.savetxt(f'{prefix}.fam', [[f'i{i}', f'i{i}', 0, 0, 0, -9] for i in range(n)], fmt='%s')


def genotypes(n=61, m=103, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, rng.uniform(0.1, 0.9, m), size=(n, m)).astype(float)
    G[rng.random(G.shape) < 0.01] = np.nan
    return G


def test_bed_roundtrip_and_subset(tmp_path):
    G = genotypes()
    write_plink(tmp_path / 'g', G)
    assert np.array_equal(gwas.read_bed(tmp_path / 'g'), G, equal_nan=True)
    keep = np.arange(len(G)) % 3 != 0
    gwas.write_bed_subset(tmp_path / 'g', keep, tmp_path / 's', chunk=10)
    np.savetxt(tmp_path / 's.fam', np.zeros((keep.sum(), 6)), fmt='%d')
    assert np.array_equal(gwas.read_bed(tmp_path / 's'), G[keep], equal_nan=True)


def test_snp_subset(tmp_path):
    G = genotypes()
    write_plink(tmp_path / 'g', G)
    snps = np.random.default_rng(2).random(G.shape[1]) < 0.7
    gwas.write_snp_subset(tmp_path / 'g', snps, tmp_path / 's')
    assert np.array_equal(gwas.read_bed(tmp_path / 's'), G[:, snps], equal_nan=True)
    bim = np.loadtxt(tmp_path / 's.bim', dtype=str)
    assert list(bim[:, 1]) == [f'snp{j}' for j in np.flatnonzero(snps)]


@pytest.mark.skipif(not Path(gwas.gemma_binary()).exists(), reason='GEMMA binary not installed')
def test_run_gemma_matches_exact_lmm(tmp_path):
    """Wald betas and standard errors at GEMMA's per-SNP variance ratio.

    With missing phenotypes and covariates.
    """
    G = genotypes()
    write_plink(tmp_path / 'g', G)
    rng = np.random.default_rng(1)
    n = len(G)
    covariates = rng.normal(size=(n, 2))
    causal = np.nan_to_num(G[:, :5] - np.nanmean(G[:, :5], 0))
    Y = causal @ rng.normal(size=(5, 2)) + rng.normal(size=(n, 2)) + 3
    Y[[2, 7, 30]] = np.nan
    result = gwas.run_gemma(tmp_path / 'g', Y, tmp_path / 'work', covariates=covariates)

    keep = ~np.isnan(Y[:, 0])
    X = gwas.read_bed(tmp_path / 'g')[keep].astype(float)
    # GEMMA: mean-imputed, centered
    X = np.where(np.isnan(X), np.nanmean(X, 0), X) - np.nanmean(X, 0)
    # SNPs passing GEMMA's filters, which also make up the relatedness
    snps = [int(s[3:]) for s in result['rs']]
    K = X[:, snps] @ X[:, snps].T / len(snps)
    W = np.hstack([np.ones((keep.sum(), 1)), covariates[keep]])
    for t in range(Y.shape[1]):
        y = Y[keep, t]
        for k, j in enumerate(snps):
            H_inv = np.linalg.inv(result['l_remle'][t, k] * K + np.eye(len(y)))
            P = H_inv - H_inv @ W @ np.linalg.solve(W.T @ H_inv @ W, W.T @ H_inv)
            x = X[:, j]
            beta = x @ P @ y / (x @ P @ x)
            sigma2 = (y @ P @ y - beta ** 2 * (x @ P @ x)) / (len(y) - W.shape[1] - 1)
            assert result['beta'][t, k] == pytest.approx(beta, rel=1e-5)
            assert result['se'][t, k] == pytest.approx(np.sqrt(sigma2 / (x @ P @ x)), rel=1e-5)


def test_line_means():
    Y = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [7.0, 6.0]])
    out = gwas.line_means(Y, [2, 2, -1, 0], n_lines=4)
    Z = (Y - Y.mean(axis=0)) / Y.std(axis=0)  # standardized over all rows, line -1 included
    np.testing.assert_allclose(out[0], Z[3])
    np.testing.assert_allclose(out[2], Z[:2].mean(axis=0))
    assert np.isnan(out[[1, 3]]).all()


def test_plink_set_rejects_wrong_fam(tmp_path):
    G = genotypes()
    write_plink(tmp_path / 'g', G)
    np.savetxt(tmp_path / 'subset.fam', [[f'i{i}', f'i{i}', 0, 0, 0, -9] for i in range(40)],
               fmt='%s')
    with pytest.raises(ValueError, match='bytes'):
        gwas.write_plink_set(tmp_path / 'g', tmp_path / 'subset.fam', tmp_path / 's')
    gwas.write_plink_set(tmp_path / 'g', tmp_path / 'g.fam', tmp_path / 's')
    assert np.array_equal(gwas.read_bed(tmp_path / 's'), G, equal_nan=True)


def test_cross_fit_takes_each_row_from_its_held_out_fold():
    from h2opt_analysis.gwas import cross_fit
    rng = np.random.default_rng(0)
    base = rng.normal(size=40)
    fold_of = np.arange(40) % 4
    traits = np.stack([base * (-1) ** f * (f + 1) + f for f in range(4)])  # sign, scale, shift
    stitched, r = cross_fit(traits, fold_of)
    np.testing.assert_allclose(r, 1)
    z = (base - base.mean()) / base.std()
    np.testing.assert_allclose(stitched, z)
