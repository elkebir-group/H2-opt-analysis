"""One permutation of the null of the whole Miscanthus hybrid H2-opt -> GWAS pipeline, one date.

The plants are permuted within reps, as the sorghum and Arabidopsis nulls (h2opt_analysis.nulls):
a plant takes the family and the genotypes of another plant of its rep; its plot, rep and column
stay with its images. On the permuted families: hybrid H2-opt on
fold 0 as train_hybrid.py (the linear noise that choose_noise.py chose for fold 0 and the CNN level
of choose_cnn_level.py; no new selection), trained on the heritability after the nearest-neighbor
field adjustment (--neighbor-width), Henderson3 heritability scored after it (slope fitted on each
split), and the GWAS of gwas.py (GEMMA of trait 1 with an intercept, the genotype PCs and the
neighbor mean), with the neighbors of the permuted families. No genotype-phenotype relation is
left, so the GWAS hits measure what the pipeline makes from nothing. Writes an .npz: the source
plant of each plant (the permutation), the traits, the GWAS phenotypes and the -log10 p of every
SNP, so that a new GWAS step needs no new training, and the summary (nulls.null_summary).

    python pipelines/scripts/miscanthus/null_gwas.py --population central --images images.npy \\
        --noise noise.npz --cnn-level cnn_level.npz --date 2 --permutation 0 \\
        --genotypes gwas/genotypes --pcs gwas/genotype_pcs.npy --out perm0.npz --work work/perm0 \\
        --linear-steps 10000 --steps 1000 --max-copies 4 --device cuda --threads 1 \\
        --neighbor-width 3
"""

import argparse
import types

import numpy as np
import torch
from h2opt import PositionImageConvModel, heritability, synthetic_traits, train_linear_conv_models
from h2opt.selection import noise_scale

from h2opt_analysis import gwas, miscanthus, nulls

FOLD = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--noise', required=True, help='choose_noise.py output')
    parser.add_argument('--cnn-level', required=True, help='choose_cnn_level.py output')
    parser.add_argument('--date', type=int, required=True, help='index into miscanthus.DATES')
    parser.add_argument('--permutation', type=int, required=True, help='seed of the permutation')
    parser.add_argument('--genotypes', required=True, help='PLINK prefix (genotypes.py)')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--linear-steps', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--max-copies', type=int, help='models trained at once (GPU memory)')
    parser.add_argument('--device', required=True, help="'cpu' or 'cuda'")
    parser.add_argument('--threads', type=int, required=True, help='OpenBLAS threads per run')
    parser.add_argument('--neighbor-width', type=int, required=True,
                        help='the nearest-neighbor field adjustment (field columns), as in '
                             'train_hybrid.py and gwas.py')
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    rows = miscanthus.genotype_rows(args.population, f'{args.genotypes}.fam')
    source = nulls.permute_genotypes(np.arange(len(design.families)), design.environment[:, 0],
                                     seed=args.permutation)
    families, rows = design.families[source], rows[source]
    weights = miscanthus.neighbor_weights(
        types.SimpleNamespace(**{**vars(design), 'families': families}), args.neighbor_width)

    torch.manual_seed(args.permutation)
    x = miscanthus.load_date(args.images, design.keep, args.date)
    flat = x.reshape((len(x), -1))
    with np.load(args.noise) as f:
        noise_sd = f['noise_sd'][FOLD, args.date]
    with np.load(args.cnn_level) as f:
        level = float(f['chosen'])
    train_test = (design.fold == FOLD).astype(int)
    conv_sd = level * noise_scale(flat[train_test == 0])
    model, = train_linear_conv_models(
        lambda: PositionImageConvModel(x.shape[1], x.shape[2:]), x, families, design.environment,
        train_test[None], np.array([[noise_sd]]), np.array([[conv_sd]]), n_traits=1,
        n_linear_iter=args.linear_steps, n_iter=args.steps, device=args.device,
        subgroups=design.plots, estimator='henderson3', max_copies=args.max_copies, verbose=False,
        neighbors=weights)
    Y = synthetic_traits(model, x, train_test == 0)

    def h2(on):
        y = miscanthus.papadakis(Y, weights, on)
        return heritability(y[on], families[on], design.environment[on], design.plots[on],
                            'henderson3')

    n_lines = len(gwas.read_fam(args.genotypes))
    phenotypes = gwas.line_means(Y, rows, n_lines)
    covariates = np.column_stack([np.load(args.pcs),
                                  miscanthus.neighbor_covariates(Y, weights, rows, n_lines)])
    result = gwas.run_gemma(args.genotypes, phenotypes, args.work, covariates=covariates,
                            jobs=1, threads=args.threads)
    scores = gwas.summarize(result)
    nulls.save(args.out, scores, h2(train_test == 0), h2(train_test == 1),
               permutation=args.permutation, source=source, train_test=train_test, traits=Y,
               phenotypes=phenotypes)


if __name__ == '__main__':
    main()
