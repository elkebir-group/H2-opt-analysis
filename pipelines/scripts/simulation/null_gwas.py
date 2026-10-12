"""One permutation of the null of the whole linear H2-opt -> GWAS pipeline on one spectral
simulation instance.

The plants are permuted as in the sorghum and Arabidopsis nulls (h2opt_analysis.nulls), with no
strata (the simulation has no environment): a plant takes the line, and so the SNPs and the fold
(simulation.line_folds), of another plant. On the permuted lines: linear H2-opt on the plants
outside fold 0 as train_h2opt.py (the noise of choose_noise.py for the real instance; no new
selection), and the GWAS of gwas.py (line means, GEMMA with an intercept and the genotype PCs) of
traits 1 to simulation.N_TRAITS. No genotype-phenotype relation is left, so the GWAS hits measure
what the pipeline makes from nothing. Writes an .npz: the source plant of each plant, the traits,
the GWAS phenotypes and the -log10 p of every SNP, and the summary (nulls.null_summary).

    python pipelines/scripts/simulation/null_gwas.py --measurements spectra.npz \\
        --lines spectra.npz --noise noise.npz --instance 0 --permutation 0 \\
        --genotypes genotypes --pcs genotype_pcs.npy --out perm0.npz --work work/perm0 \\
        --steps 10000 --device cuda --jobs 3 --threads 1
"""

import argparse

import numpy as np
from h2opt import LinearModel

from h2opt_analysis import gwas, nulls, simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', required=True, help='.npz with X, names (plants first)')
    parser.add_argument('--lines', required=True, help="spectra.npz of the instance ('lines')")
    parser.add_argument('--noise', required=True, help='choose_noise.py output')
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--permutation', type=int, required=True, help='seed of the permutation')
    parser.add_argument('--genotypes', required=True, help='PLINK prefix')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--steps', type=int, required=True, help='steps per trait')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    parser.add_argument('--jobs', type=int, required=True, help='traits run at once')
    parser.add_argument('--threads', type=int, required=True, help='OpenBLAS threads per run')
    args = parser.parse_args()

    with np.load(args.measurements) as f:
        X, names = f['X'], f['names']
    X = X.reshape((len(X), -1))
    with np.load(args.lines) as f:
        lines = f['lines']
    with np.load(args.noise) as f:
        noise_sd = float(f['noise_sd'])
    source = nulls.permute_genotypes(np.arange(len(names)), seed=args.permutation)
    train_test = (simulation.instance_folds(args.instance)[source] == 0).astype(int)
    Y, h2_train, h2_test = nulls.train_null_traits(
        lambda: LinearModel(X.shape[1]), X, names[source], None, train_test,
        simulation.N_TRAITS, noise_sd, args.steps, None, args.device, seed=args.permutation)

    phenotypes = gwas.line_means(Y, lines[source], len(gwas.read_fam(args.genotypes)))
    result = gwas.run_gemma(args.genotypes, phenotypes, args.work, covariates=np.load(args.pcs),
                            jobs=args.jobs, threads=args.threads)
    scores = gwas.summarize(result)
    summary = nulls.null_summary(scores, h2_train, h2_test)
    np.savez(args.out, permutation=args.permutation, source=source, train_test=train_test,
             traits=Y, phenotypes=phenotypes, chr=scores['chr'], ps=scores['ps'],
             neg_log10_p=scores['neg_log10_p'], **summary)
    for t in range(simulation.N_TRAITS):
        print(f'trait {t + 1}: h2 train {summary["h2_train"][t]:.3f} test '
              f'{summary["h2_test"][t]:.3f}, {summary["hits"][t]} Bonferroni hits, max -log10 p '
              f'{summary["max_neg_log10_p"][t]:.2f}, lambda GC {summary["lambda_gc"][t]:.3f}',
              flush=True)


if __name__ == '__main__':
    main()
