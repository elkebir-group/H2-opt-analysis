"""One permutation of the null of the whole Arabidopsis H2-opt -> GWAS pipeline, one treatment.

The accession labels of the samples are permuted within years (h2opt_analysis.nulls), so a sample
takes the accession, and so the SNPs, of another sample of its year; its year stays. On the
permuted labels: the accession folds, linear H2-opt on fold 0 as train_h2opt.py (the noise that
choose_noise.py chose for fold 0 of the real data, and the config's penalty; no new selection),
and the GWAS of gwas.py (accession means, GEMMA with an intercept and the genotype PCs). No
genotype-phenotype relation is left, so the GWAS hits measure what the pipeline makes from
nothing. Writes an .npz (nulls.null_summary).

    python pipelines/scripts/arabidopsis/null_gwas.py --features features.npy --treatment 6d \\
        --permutation 0 --noise noise.npz --genotypes gwas/genotypes --pcs gwas/genotype_pcs.npy \\
        --out perm0.npz --work work/perm0 --traits 5 --steps 16000 --penalty l1 1e-6 \\
        --device cuda --jobs 5 --threads 1
"""

import argparse

import numpy as np
from h2opt import LinearModel

from h2opt_analysis import arabidopsis, gwas, nulls

FOLD = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='.npy of features.py')
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
    parser.add_argument('--permutation', type=int, required=True, help='seed of the permutation')
    parser.add_argument('--noise', required=True, help='the noise chosen by choose_noise.py')
    parser.add_argument('--genotypes', required=True, help='PLINK prefix (genotypes.py)')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True, help='steps per trait')
    parser.add_argument('--penalty', nargs=2, required=True, metavar=('KIND', 'WEIGHT'),
                        help="penalty of h2opt.train: 'l1', 'l2' or 'tv', and its weight")
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    parser.add_argument('--jobs', type=int, required=True, help='traits run at once')
    parser.add_argument('--threads', type=int, required=True, help='OpenBLAS threads per run')
    args = parser.parse_args()

    X, accessions, environment = arabidopsis.load(args.features, args.treatment)
    permuted = nulls.permute_genotypes(accessions, environment, seed=args.permutation)
    train_test = (arabidopsis.folds(permuted) == FOLD).astype(int)
    with np.load(args.noise) as f:
        noise_sd = f['noise_sd'][list(f['folds']).index(FOLD)]
    Y, h2_train, h2_test = nulls.train_null_traits(
        lambda: LinearModel(X.shape[1]), X, permuted, environment, train_test, args.traits,
        noise_sd, args.steps, (args.penalty[0], float(args.penalty[1])), args.device,
        seed=args.permutation)

    rows = arabidopsis.genotype_rows(permuted, f'{args.genotypes}.fam')
    phenotypes = gwas.line_means(Y, rows, len(gwas.read_fam(args.genotypes)))
    result = gwas.run_gemma(args.genotypes, phenotypes, args.work, covariates=np.load(args.pcs),
                            jobs=args.jobs, threads=args.threads)
    summary = nulls.null_summary(gwas.summarize(result), h2_train, h2_test)
    np.savez(args.out, permutation=args.permutation, **summary)
    for t in range(args.traits):
        print(f'trait {t + 1}: h2 train {summary["h2_train"][t]:.3f} test '
              f'{summary["h2_test"][t]:.3f}, {summary["hits"][t]} Bonferroni hits, max -log10 p '
              f'{summary["max_neg_log10_p"][t]:.2f}, lambda GC {summary["lambda_gc"][t]:.3f}',
              flush=True)


if __name__ == '__main__':
    main()
