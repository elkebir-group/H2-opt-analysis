"""Permuted-genotype null of the whole H2-opt -> GWAS pipeline.

Each measured individual (row) is attributed to the genotype of another row, so no
genotype-phenotype relation remains while the genotype structure (group sizes per stratum) is kept.
The permuted labels must be computed once, on all rows, and used everywhere a genotype label is
used: training H2-opt, the folds, heritability evaluation and the per-genotype phenotypes given to
GWAS (which then pairs them with the real SNPs of the permuted label). Environment stays attached to
the row. Subset rows (an environment, a fold) after permuting.
"""

import numpy as np

from .gwas import bonferroni_hits


def permute_genotypes(genotypes, strata=None, seed=0):
    """Genotype labels permuted among the rows of each stratum (e.g. field set or treatment).

    strata: (n,) or (n, k) array; rows with equal values form a stratum. None permutes across all
    rows."""
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


def train_null_traits(make_model, X, groups, environment, train_test, n_traits, noise_sd, n_iter,
                      penalty, device, seed, estimator='anova', subgroups=None):
    """H2-opt traits of one permuted dataset, trained as the real ones (same model, noise and
    penalty; no new selection). groups: the permuted genotype labels. make_model: a function of no
    arguments that returns the model of one trait (e.g. a LinearModel). Returns the traits of all
    rows ((n, n_traits), uncorrelated on the training rows) and the train and test heritability of
    each trait under the permuted labels."""
    import torch
    from h2opt import TraitModels, heritability, synthetic_traits, train_models

    torch.manual_seed(seed)
    train_test = np.asarray(train_test)
    model = TraitModels(n_traits, make_model)
    train_models([model], X, groups, environment, train_test[None], n_traits=n_traits,
                 n_iter=n_iter, noise_sd=noise_sd, penalty=penalty, device=device,
                 estimator=estimator, subgroups=subgroups, verbose=False)
    Y = synthetic_traits(model, X, train_test == 0)

    def h2(rows):
        return heritability(Y[rows], np.asarray(groups)[rows],
                            None if environment is None else np.asarray(environment)[rows],
                            None if subgroups is None else np.asarray(subgroups)[rows], estimator)

    return Y, h2(train_test == 0), h2(train_test == 1)


def null_summary(summary, h2_train, h2_test):
    """What a permutation null records of its GWAS summary (gwas.summarize): the Bonferroni hits,
    the largest -log10 p and lambda GC of each trait, and the train and test heritability."""
    score = summary['neg_log10_p']
    return {'hits': bonferroni_hits(score), 'max_neg_log10_p': score.max(axis=1),
            'lambda_gc': summary['lambda_gc'], 'h2_train': np.asarray(h2_train),
            'h2_test': np.asarray(h2_test)}


def save(out, scores, h2_train, h2_test, **arrays):
    """Save one permutation of a null to the .npz out and print its summary per trait. Saves the
    arrays (e.g. the permuted labels, the folds, the traits and the GWAS phenotypes), the -log10 p
    of every SNP of scores (gwas.summarize) and null_summary."""
    summary = null_summary(scores, h2_train, h2_test)
    np.savez(out, **arrays, chr=scores['chr'], ps=scores['ps'],
             neg_log10_p=scores['neg_log10_p'], **summary)
    for t in range(len(summary['hits'])):
        print(f'trait {t + 1}: h2 train {summary["h2_train"][t]:.3f} test '
              f'{summary["h2_test"][t]:.3f}, {summary["hits"][t]} Bonferroni hits, max -log10 p '
              f'{summary["max_neg_log10_p"][t]:.2f}, lambda GC {summary["lambda_gc"][t]:.3f}',
              flush=True)
