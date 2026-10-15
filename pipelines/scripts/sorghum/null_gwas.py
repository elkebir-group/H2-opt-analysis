"""One permutation of the null of the whole sorghum H2-opt -> GWAS pipeline.

The genotype labels of the plants are permuted within field sets (h2opt_analysis.nulls), so a
plant takes the line, and so the SNPs, of another plant of its set; its field set stays. On the
permuted labels: the genotype folds, linear H2-opt on fold 0 as train_h2opt.py (the noise that
choose_noise.py chose for fold 0 of the real data, and the config's penalty; no new selection),
and the GWAS of gwas.py (line means, GEMMA with an intercept and the genotype PCs). No genotype-
phenotype relation is left, so the GWAS hits measure what the pipeline makes from nothing. Writes
an .npz: the permuted labels, the traits, the GWAS phenotypes and the -log10 p of every SNP, so
that a new GWAS step needs no new training, and the summary (nulls.null_summary: Bonferroni hits,
max -log10 p and lambda GC per trait, train and test heritability under the permuted labels).

    python pipelines/scripts/sorghum/null_gwas.py --permutation 0 --noise noise.npz \\
        --pcs genotype_pcs.npy --out perm0.npz --work work/perm0 --traits 5 --steps 320000 \\
        --penalty tv 1e-5 --device cuda --jobs 3 --threads 1
"""

import argparse

import numpy as np
from h2opt import LinearModel

from h2opt_analysis import gwas, nulls, sorghum

FOLD = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--permutation', type=int, required=True, help='seed of the permutation')
    parser.add_argument('--noise', required=True, help='the noise chosen by choose_noise.py')
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

    X, genotypes, environment = sorghum.load()
    permuted = nulls.permute_genotypes(genotypes, environment, seed=args.permutation)
    train_test = (sorghum.folds(permuted) == FOLD).astype(int)
    with np.load(args.noise) as f:
        noise_sd = f['noise_sd'][list(f['folds']).index(FOLD)]
    Y, h2_train, h2_test = nulls.train_null_traits(
        lambda: LinearModel(X.shape[1]), X, permuted, environment, train_test, args.traits,
        noise_sd, args.steps, (args.penalty[0], float(args.penalty[1])), args.device,
        seed=args.permutation)

    n_lines = len(gwas.read_fam(sorghum.GENOTYPES))
    phenotypes = gwas.line_means(Y, sorghum.line_of_plant(permuted), n_lines)
    result = gwas.run_gemma(sorghum.GENOTYPES, phenotypes, args.work, covariates=np.load(args.pcs),
                            jobs=args.jobs, threads=args.threads)
    scores = gwas.summarize(result)
    nulls.save(args.out, scores, h2_train, h2_test, permutation=args.permutation, labels=permuted,
               train_test=train_test, traits=Y, phenotypes=phenotypes)


if __name__ == '__main__':
    main()
