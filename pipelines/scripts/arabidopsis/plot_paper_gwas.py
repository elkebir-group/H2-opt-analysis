"""GWAS of the H2-opt traits of one Arabidopsis treatment over all folds (images/metab0d_GWAS,
metab6d_GWAS), one row per trait with a significant SNP, in the style of sorGWAS.

The regions are those of the paper's gene table: the Bonferroni hits (0.05 / SNPs) of all traits
and folds, grouped by LD clumping (gwas.clump: r^2 above --r2 with the lead SNP within --kb
kilobases, LD from --genotypes), and then loci whose lead SNPs are on the same chromosome and
within --kb kilobases of each other joined into one region. The lead SNP of a region is its most
significant SNP.

Each row has a QQ plot (left: one curve per fold, colored by fold, and the diagonal) and a
Manhattan plot (right: the largest -log10 p over the folds of each SNP in gray, the SNPs above the
threshold colored by the fold with the largest -log10 p, and the threshold as a dotted line). A
circle marks, for each region in which the trait has a significant SNP, the trait's strongest SNP
of the region. Regions of more than one locus are labeled by chromosome and the position of the
region's lead SNP in Mb (TAIR8 positions of the SNP panel; they agree with TAIR10 to 0.01 Mb at
the lead SNPs). Each row has its own y range. Folds are
numbered from 1, as in the paper. Saves --out (.pdf) and a .png next to it, and prints the
regions.

    python pipelines/scripts/arabidopsis/plot_paper_gwas.py --gwas 0d/gwas/fold{0..4}.npz \\
        --genotypes 0d/gwas/genotypes --r2 0.5 --kb 250 --out metab0d_GWAS.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from h2opt_analysis import gwas  # noqa: E402
from h2opt_analysis.figures import (  # noqa: E402
    GWAS_LABEL_SIZE,
    GWAS_ROW_HEIGHT,
    MATH_RC,
    SIM_FIG_RC,
    manhattan_x,
    qq_curve,
)

FOLD_COLORS = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple')
GRAYS = ('0.55', '0.75')  # alternating chromosomes
MARGINS = 0.55  # inches above and below the rows (legend, x labels)


def regions(loci, chrom, ps, score, kb):
    """Loci joined into regions: loci whose lead SNPs are on the same chromosome and within kb
    kilobases of each other (single linkage, by position). Each region: snps (all SNP indices of
    its loci), loci (count) and lead (its SNP with the largest -log10 p over all rows)."""
    best = score.max(axis=0)
    leads = sorted(loci, key=lambda locus: (chrom[locus['lead']], ps[locus['lead']]))
    out = []
    for locus in leads:
        lead = locus['lead']
        last = out[-1] if out else None
        if last and chrom[last['last']] == chrom[lead] \
                and ps[lead] - ps[last['last']] <= kb * 1000:
            last['snps'] = np.concatenate([last['snps'], locus['snps']])
            last['loci'] += 1
            last['last'] = lead
        else:
            out.append({'snps': np.asarray(locus['snps']), 'loci': 1, 'last': lead})
    for region in out:
        region['lead'] = region['snps'][np.argmax(best[region['snps']])]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gwas', nargs='+', required=True, help='gwas.summarize .npz per fold')
    parser.add_argument('--genotypes', required=True, help='PLINK prefix of the GWAS genotypes')
    parser.add_argument('--r2', type=float, required=True)
    parser.add_argument('--kb', type=float, required=True)
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    args = parser.parse_args()

    folds = [np.load(path) for path in args.gwas]
    chrom, ps = folds[0]['chr'], folds[0]['ps']
    for f in folds[1:]:
        assert np.array_equal(f['chr'], chrom) and np.array_equal(f['ps'], ps), \
            'the folds have different SNPs'
    score = np.stack([f['neg_log10_p'] for f in folds])  # folds x traits x SNPs
    n_folds, n_traits, n_snps = score.shape
    threshold = -np.log10(0.05 / n_snps)
    loci = gwas.clump(args.genotypes, chrom, ps, score.reshape(-1, n_snps), r2=args.r2,
                      kb=args.kb)
    joined = regions(loci, chrom, ps, score.reshape(-1, n_snps), args.kb)
    hits = score > threshold
    print(f'{n_snps} SNPs, threshold -log10 p {threshold:.2f}; {len(loci)} loci in '
          f'{len(joined)} regions')
    for region in joined:
        snps = region['snps']
        traits = [t + 1 for t in range(n_traits) if hits[:, t][:, snps].any()]
        in_folds = [f + 1 for f in range(n_folds) if hits[f][:, snps].any()]
        lead = region['lead']
        print(f'  chr {chrom[lead]}, {ps[snps].min() / 1e6:.2f}-{ps[snps].max() / 1e6:.2f} Mb: '
              f'{region["loci"]} loci, lead {ps[lead]:,} (-log10 p '
              f'{score[:, :, lead].max():.2f}), traits {traits}, folds {in_folds}')

    rows = [t for t in range(n_traits) if hits[:, t].any()]
    x, order, centers, length = manhattan_x(chrom, ps)
    position = {c: k for k, c in enumerate(order)}
    parity = np.array([position[c] for c in chrom]) % 2
    best = score.max(axis=0)  # traits x SNPs
    best_fold = score.argmax(axis=0)
    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig, axes = plt.subplots(len(rows), 2, figsize=(6.5, GWAS_ROW_HEIGHT * len(rows)
                                                        + MARGINS),
                                 gridspec_kw={'width_ratios': [1, 4]}, squeeze=False,
                                 sharex='col')
        for t, (qq, ax) in zip(rows, axes, strict=True):
            top = max(best[t].max() * 1.25, threshold * 1.4)
            for f in range(n_folds):
                end = qq_curve(qq, score[f, t], FOLD_COLORS[f], 0.6, 3, 0.8)
            qq.plot([0, end], [0, end], color='black', linewidth=0.6, linestyle='--', zorder=1)
            qq.axhline(threshold, color='black', linestyle=':', linewidth=0.8)
            qq.set_ylim(0, top)
            qq.xaxis.set_major_locator(MultipleLocator(2))
            qq.set_ylabel(r'$-\log_{10} p$')
            qq.spines[['top', 'right']].set_visible(False)
            ax.scatter(x, best[t], s=1, c=np.array(GRAYS)[parity], linewidths=0, rasterized=True)
            significant = np.flatnonzero(best[t] > threshold)
            ax.scatter(x[significant], best[t, significant], s=3, linewidths=0, zorder=3,
                       c=np.array(FOLD_COLORS)[best_fold[t, significant]])
            ax.axhline(threshold, color='black', linestyle=':', linewidth=0.8)
            labeled = []
            for region in joined:
                members = region['snps'][best[t, region['snps']] > threshold]
                if not len(members):
                    continue
                mark = members[np.argmax(best[t, members])]
                lead = region['lead']
                ax.scatter(x[mark], best[t, mark], s=16, facecolors='none', edgecolors='black',
                           linewidths=0.6, zorder=4)
                if region['loci'] == 1:
                    continue
                # left of the circle, or right of it when a labeled circle is close on the left
                right = any(0 < x[mark] - other < 0.12 * length for other in labeled)
                labeled.append(x[mark])
                ax.annotate(f'{chrom[lead]}: {ps[lead] / 1e6:.2f} Mb', (x[mark], best[t, mark]),
                            xytext=(4 if right else -4, 0), textcoords='offset points',
                            ha='left' if right else 'right', va='center',
                            fontsize=GWAS_LABEL_SIZE)
            ax.set_xlim(0, length)
            ax.set_ylim(0, top)
            ax.set_ylabel(r'$-\log_{10} p$')
            ax.spines[['top', 'right']].set_visible(False)
            ax.text(1.005, 0.5, f'trait {t + 1}', transform=ax.transAxes, rotation=270,
                    va='center', ha='left', fontweight='bold')
        axes[-1, 0].set_xlabel(r'expected $-\log_{10} p$')
        axes[-1, 1].set_xticks(centers, [str(c) for c in order])
        axes[-1, 1].set_xlabel('chromosome')
        handles = [Line2D([], [], color=c, linewidth=1.2) for c in FOLD_COLORS[:n_folds]]
        fig.legend(handles, [f'fold {f + 1}' for f in range(n_folds)], loc='upper center',
                   ncol=n_folds, frameon=False, handlelength=1.2, columnspacing=1.0,
                   borderaxespad=0.1)
        fig.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.6, rect=(0, 0, 0.98, 1 - 0.18 /
                                                              fig.get_size_inches()[1]))
        fig.savefig(args.out)
        fig.savefig(args.out.removesuffix('.pdf') + '.png', dpi=300)


if __name__ == '__main__':
    main()
