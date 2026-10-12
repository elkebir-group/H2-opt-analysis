"""One permutation of the null of the whole hybrid H2-opt -> GWAS pipeline on one image simulation
instance.

As pipelines/scripts/simulation/null_gwas.py (plants permuted with no strata: a plant takes the
line, the SNPs and the fold of another plant; GWAS of traits 1 to simulation.N_TRAITS), with the
hybrid of train_hybrid.py at the instance's fixed levels (fixed_levels.py) on the plants outside
fold 0. Writes an .npz: the source plant of each plant, the traits, the GWAS phenotypes and the
-log10 p of every SNP, and the summary (nulls.null_summary).

    python pipelines/scripts/image_simulation/null_gwas.py --measurements images.npz \\
        --lines spectra.npz --noise noise.npz --cnn-level cnn_level.npz --instance 0 \\
        --permutation 0 --genotypes genotypes --pcs genotype_pcs.npy --out perm0.npz \\
        --work work/perm0 --linear-steps 10000 --steps 1000 --max-copies 4 --device cuda \\
        --jobs 3 --threads 1
"""

import argparse

import numpy as np
import torch
from h2opt import heritability, synthetic_traits, train_linear_conv_models
from h2opt.selection import noise_scale

from h2opt_analysis import gwas, nulls, simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', required=True, help='.npz with X, names (plants first)')
    parser.add_argument('--lines', required=True, help="spectra.npz of the instance ('lines')")
    parser.add_argument('--noise', required=True, help='fixed_levels.py linear level')
    parser.add_argument('--cnn-level', required=True, help='fixed_levels.py CNN level')
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--permutation', type=int, required=True, help='seed of the permutation')
    parser.add_argument('--genotypes', required=True, help='PLINK prefix')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--linear-steps', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max-copies', type=int, help='models trained at once (GPU memory)')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    parser.add_argument('--jobs', type=int, required=True, help='traits run at once')
    parser.add_argument('--threads', type=int, required=True, help='OpenBLAS threads per run')
    args = parser.parse_args()

    torch.manual_seed(args.permutation)
    with np.load(args.measurements) as f:
        X, names = f['X'], f['names']
    with np.load(args.lines) as f:
        lines = f['lines']
    with np.load(args.noise) as f:
        noise_sd = float(f['noise_sd'])
    with np.load(args.cnn_level) as f:
        cnn_level = float(f['chosen'])
    source = nulls.permute_genotypes(np.arange(len(names)), seed=args.permutation)
    groups = names[source]
    test = simulation.instance_folds(args.instance)[source] == 0
    model, = train_linear_conv_models(
        simulation.make_cnn(X), X, groups, None, test.astype(int)[None], noise_sd,
        cnn_level * noise_scale(X[~test].reshape(((~test).sum(), -1))),
        n_traits=simulation.N_TRAITS, n_linear_iter=args.linear_steps, n_iter=args.steps,
        device=args.device, max_copies=args.max_copies, verbose=False)
    Y = synthetic_traits(model, X, ~test)

    def h2(rows):
        return heritability(Y[rows], groups[rows], None, None, 'anova')

    phenotypes = gwas.line_means(Y, lines[source], len(gwas.read_fam(args.genotypes)))
    result = gwas.run_gemma(args.genotypes, phenotypes, args.work, covariates=np.load(args.pcs),
                            jobs=args.jobs, threads=args.threads)
    scores = gwas.summarize(result)
    summary = nulls.null_summary(scores, h2(~test), h2(test))
    np.savez(args.out, permutation=args.permutation, source=source, train_test=test.astype(int),
             traits=Y, phenotypes=phenotypes, chr=scores['chr'], ps=scores['ps'],
             neg_log10_p=scores['neg_log10_p'], **summary)
    for t in range(simulation.N_TRAITS):
        print(f'trait {t + 1}: h2 train {summary["h2_train"][t]:.3f} test '
              f'{summary["h2_test"][t]:.3f}, {summary["hits"][t]} Bonferroni hits, max -log10 p '
              f'{summary["max_neg_log10_p"][t]:.2f}, lambda GC {summary["lambda_gc"][t]:.3f}',
              flush=True)


if __name__ == '__main__':
    main()
