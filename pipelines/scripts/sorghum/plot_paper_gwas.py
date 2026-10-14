"""GWAS figure of the paper (sorGWAS): the GWAS of the sorghum synthetic traits with significant
SNPs, per fold and cross-fitted.

The loci are the Bonferroni hits (0.05 / SNPs) of all traits and folds, grouped by LD clumping
(gwas.clump: r^2 above --r2 with the lead SNP within --kb kilobases, LD from --genotypes). For each
trait with a significant SNP in some fold, one row per such fold and then a row of the trait's
cross-fitted GWAS (bold label). Each row has a QQ plot (left, observed against expected -log10 p,
with lambda GC) and a Manhattan plot (right; dotted line: the Bonferroni threshold). In a fold
row, a circle marks the strongest significant SNP of each locus that the trait hits in that fold;
in the cross-fit row, the SNP of each of the trait's loci with the largest cross-fitted -log10 p.
Each locus is labeled by chromosome and the position of its lead SNP. Folds are numbered from 1,
as in the paper. Saves --out (.pdf) and a .png next to it, and prints the loci.

    python pipelines/scripts/sorghum/plot_paper_gwas.py --gwas h2opt_fold{0..9}.npz \\
        --crossfit h2opt_crossfit_trait{1..10}.npz --genotypes WEST_original --r2 0.5 --kb 250 \\
        --out figures/paper/sorGWAS.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis import gwas  # noqa: E402
from h2opt_analysis.figures import MATH_RC, SIM_FIG_RC, letter  # noqa: E402

ROW_HEIGHT = 0.46  # inches
LABEL_SIZE = 5.5
LABEL_STEP = 6  # points between stacked locus labels


def manhattan_x(chrom, ps):
    """Position of each SNP on one axis with the chromosomes in sorted order, the center of each
    chromosome and the axis length."""
    order = np.unique(chrom)
    offset = np.zeros(len(ps), dtype=np.int64)
    centers, start = [], 0
    for c in order:
        on = chrom == c
        offset[on] = start
        centers.append(start + (ps[on].max() + 1) / 2)
        start += ps[on].max() + 1
    return ps + offset, order, centers, start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gwas', nargs='+', required=True, help='gwas.py output of each fold')
    parser.add_argument('--crossfit', nargs='+', required=True,
                        help='cross-fitted GWAS of traits 1, 2, ... (one .npz each)')
    parser.add_argument('--genotypes', required=True, help='PLINK set (prefix) for the LD')
    parser.add_argument('--r2', type=float, required=True)
    parser.add_argument('--kb', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    folds = [np.load(path) for path in args.gwas]
    chrom, ps = folds[0]['chr'], folds[0]['ps']
    for f in folds[1:]:
        assert np.array_equal(f['ps'], ps), 'the folds have different SNPs'
    score = np.stack([f['neg_log10_p'] for f in folds])  # folds x traits x SNPs
    n_folds, n_traits, n_snps = score.shape
    threshold = -np.log10(0.05 / n_snps)
    loci = gwas.clump(args.genotypes, chrom, ps, score.reshape(-1, n_snps), r2=args.r2,
                      kb=args.kb)
    labels = [f'{chrom[locus["lead"]]}: {ps[locus["lead"]] / 1e6:.2f} Mb' for locus in loci]
    print(f'{len(loci)} loci (LD clumping, r^2 > {args.r2} within {args.kb} kb):')
    for locus, label in zip(loci, labels, strict=True):
        rows = [f'fold {r // n_traits + 1} trait {r % n_traits + 1}' for r in locus['rows']]
        print(f'  {label} (lead {ps[locus["lead"]]}, -log10 p '
              f'{score.reshape(-1, n_snps)[:, locus["lead"]].max():.2f}): {", ".join(rows)}')

    # rows of the figure: (trait, fold or None for the cross-fit)
    panels = []
    for t in range(n_traits):
        hit_folds = [f for f in range(n_folds) if (score[f, t] > threshold).any()]
        if hit_folds:
            panels += [(t, f) for f in hit_folds] + [(t, None)]
    x, order, centers, length = manhattan_x(chrom, ps)
    parity = np.searchsorted(order, chrom) % 2

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig, axes = plt.subplots(len(panels), 2, figsize=(6.5, ROW_HEIGHT * len(panels) + 0.35),
                                 gridspec_kw={'width_ratios': [1, 4]}, squeeze=False)
        lettered = []
        for i, ((t, f), (qq, ax)) in enumerate(zip(panels, axes, strict=True)):
            last = i == len(panels) - 1
            if f is None:
                with np.load(args.crossfit[t]) as z:
                    assert np.array_equal(z['ps'], ps), 'the cross-fit has different SNPs'
                    s, lam = z['neg_log10_p'][0], z['lambda_gc'][0]
            else:
                s, lam = score[f, t], folds[f]['lambda_gc'][t]
            ax.scatter(x, s, s=1, c=np.where(parity, 'tab:red', 'tab:blue'), rasterized=True)
            ax.axhline(threshold, color='black', linestyle=':', linewidth=0.8)
            marks = []
            for locus, label in zip(loci, labels, strict=True):
                trait_rows = [r for r in locus['rows'] if r % n_traits == t]
                if not trait_rows or (f is not None and f * n_traits + t not in trait_rows):
                    continue
                members = locus['snps']
                if f is not None:
                    members = members[s[members] > threshold]
                marks.append((members[np.argmax(s[members])], label))
            # labels of marks close together are stacked downward, the highest mark first
            placed = []
            for j, label in sorted(marks, key=lambda m: -s[m[0]]):
                ax.scatter(x[j], s[j], s=14, facecolors='none', edgecolors='black',
                           linewidths=0.6)
                shift = -LABEL_STEP * sum(abs(x[j] - xp) < 0.03 * length for xp in placed)
                placed.append(x[j])
                ax.annotate(label, (x[j], s[j]), xytext=(-4, shift), textcoords='offset points',
                            ha='right', va='center', fontsize=LABEL_SIZE,
                            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.3,
                                  'alpha': 0.8})
            ax.set_xticks(centers, [str(c) for c in order] if last else [])
            ax.tick_params(axis='x', length=0)
            ax.set_xlim(0, length)
            ax.set_ylim(0, 9.5)
            ax.set_ylabel(r'$-\log_{10} p$')
            ax.spines[['top', 'right']].set_visible(False)
            name = 'cross-fit' if f is None else f'fold {f + 1}'
            ax.text(1.01, 0.5, f'trait {t + 1}\n{name}', transform=ax.transAxes, rotation=270,
                    va='center', ha='left', fontsize=6.5,
                    fontweight='bold' if f is None else 'normal')

            observed = np.sort(s)
            expected = -np.log10(np.arange(len(observed), 0, -1) / (len(observed) + 1))
            qq.scatter(expected, observed, s=2, color='tab:blue', rasterized=True)
            top = min(expected.max(), observed.max())
            qq.plot([0, top], [0, top], color='black', linewidth=0.6, linestyle='--', zorder=1)
            qq.text(0.04, 0.96, rf'$\lambda_\mathrm{{GC}}$ = {lam:.3f}', transform=qq.transAxes,
                    va='top', ha='left')
            qq.set_xlim(0, None)
            qq.set_ylim(0, 9.5)
            qq.set_ylabel(r'$-\log_{10} p$')
            qq.spines[['top', 'right']].set_visible(False)
            if not last:
                qq.set_xticklabels([])
            if t not in lettered:
                lettered.append(t)
                letter(qq, 'abcdefghij'[len(lettered) - 1], -0.45)
        axes[-1, 1].set_xlabel('chromosome')
        axes[-1, 0].set_xlabel(r'expected $-\log_{10} p$')
        fig.subplots_adjust(left=0.07, right=0.95, top=0.985, bottom=0.05, hspace=0.3,
                            wspace=0.25)
        fig.savefig(args.out)
        fig.savefig(args.out.removesuffix('.pdf') + '.png', dpi=200)
        plt.close(fig)
    print('panels:', ', '.join(f'trait {t + 1} ' + ('cross-fit' if f is None else f'fold {f + 1}')
                               for t, f in panels))


if __name__ == '__main__':
    main()
