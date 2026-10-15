"""Cross-fitted trait --trait of H2-opt: each individual's value from the fold model that held it
out (h2opt_analysis.gwas.cross_fit), so no GWAS phenotype comes from a model that trained on that
individual. A later trait need not match across folds: check 'agreement' before using it. Writes
an .npz with 'traits' in the layout that the dataset's gwas.py reads (sorghum, Arabidopsis:
individuals x 1; Miscanthus: 1 x plants x 1, one "fold") and 'agreement', the |r| of each fold's
trait with fold 0's.

    python pipelines/scripts/shared/cross_fit_traits.py --dataset sorghum --trait 2 \\
        --traits fold0.npz ... fold9.npz --out crossfit_trait2.npz
    python pipelines/scripts/shared/cross_fit_traits.py --dataset arabidopsis --treatment 6d \\
        --trait 1 --traits fold0.npz ... fold4.npz --out crossfit.npz
    python pipelines/scripts/shared/cross_fit_traits.py --dataset miscanthus \\
        --population central --trait 1 --traits date2.npz --out crossfit_date2.npz
"""

import argparse

import numpy as np

from h2opt_analysis import arabidopsis, gwas, miscanthus, sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['sorghum', 'arabidopsis', 'miscanthus'],
                        required=True)
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, help='Arabidopsis')
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, help='Miscanthus')
    parser.add_argument('--trait', type=int, required=True, help='trait (1-based)')
    parser.add_argument('--traits', nargs='+', required=True,
                        help='the fold files in fold order (Miscanthus: the one date file)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    t = args.trait - 1
    if args.dataset == 'sorghum':
        _, genotypes, _ = sorghum.load()
        fold_of = sorghum.folds(genotypes)
        traits = [sorghum.load_traits(path)[:, t] for path in args.traits]
    elif args.dataset == 'arabidopsis':
        _, accessions, _ = arabidopsis.design(args.treatment)
        fold_of = arabidopsis.folds(accessions)
        traits = [np.load(path)['traits'][:, t] for path in args.traits]
    else:
        fold_of = miscanthus.design(args.population).fold
        traits = np.load(args.traits[0])['traits'][:, :, t]
    if len(traits) != fold_of.max() + 1:
        parser.error(f'give one traits file per fold ({fold_of.max() + 1})')
    stitched, agreement = gwas.cross_fit(np.stack(traits), fold_of)
    out = stitched[None, :, None] if args.dataset == 'miscanthus' else stitched[:, None]
    np.savez(args.out, traits=out, agreement=agreement)
    print(f"|r| of each fold's trait {args.trait} with fold 0's:", np.round(agreement, 3))


if __name__ == '__main__':
    main()
