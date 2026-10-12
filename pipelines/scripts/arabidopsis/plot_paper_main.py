"""Main Arabidopsis figure of the paper (Fig. arabiResult, images/metabFig), panels a to d.

a: the binned MS1 input matrix of the first sample of each treatment (log scale; empty bins
white). b: test heritability per trait of H2-opt and every baseline, mean over the folds, one panel
per treatment. c: coefficients of traits 1-3 of the first treatment's model (MS1 only, as in the
paper), the ones with magnitude at least 2% of the trait's largest, the marker area proportional
to the magnitude. d: SNPs above the Bonferroni threshold 0.05 / SNPs, per trait and treatment.
Panel e (pathway enrichment) is not made: its input (the mummichog files) is missing.

Saves --out (.pdf) and a .png next to it.

    python pipelines/scripts/arabidopsis/plot_paper_main.py --features features.npy \\
        --treatments 0d 6d --heritability 0d/heritability.npz 6d/heritability.npz \\
        --coefficients 0d/h2opt/coefficients_fold0.npz \\
        --gwas 0d/gwas/fold0.npz 6d/gwas/fold0.npz --out metabFig.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from h2opt_analysis import arabidopsis  # noqa: E402
from h2opt_analysis.figures import COLORS  # noqa: E402
from h2opt_analysis.methods import LABELS, ordered  # noqa: E402

# Minutes per retention-time bin.
RT_BIN = 0.5
# Smallest coefficient magnitude shown, as a fraction of the trait's largest.
SHOWN = 0.02
# Smallest m/z shown in panel a (all bins below it are empty).
MZ_SHOWN = 100


def treatment_label(treatment):
    """'0d' -> '0 day'."""
    return f'{treatment.removesuffix("d")} day'


def save(fig, path):
    """Save path (.pdf) and a .png next to it."""
    fig.savefig(path)
    fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)


def significant_snps(path):
    """SNPs per trait above the Bonferroni threshold 0.05 / SNPs of a gwas.summarize .npz."""
    score = np.load(path)['neg_log10_p']
    return (score > -np.log10(0.05 / score.shape[1])).sum(axis=1)


def panel_label(ax, letter):
    ax.text(-0.3, 1.08, letter, transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='bottom')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--treatments', nargs='+', required=True)
    parser.add_argument('--heritability', nargs='+', required=True, help='one per treatment')
    parser.add_argument('--coefficients', required=True,
                        help='coefficients.py output of the first treatment')
    parser.add_argument('--gwas', nargs='+', required=True, help='one per treatment')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if not len(args.treatments) == len(args.heritability) == len(args.gwas):
        parser.error('give one heritability and one GWAS file per treatment')

    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(10, 3.4))
    grid = fig.add_gridspec(len(args.treatments), 5, width_ratios=[1, 1.4, 1.5, 0.6, 0.6],
                            left=0.05, right=0.99, top=0.84, bottom=0.12, wspace=0.55,
                            hspace=0.55)

    # a: input matrix of one sample per treatment
    features = np.load(args.features, mmap_mode='r')
    _, treatments, _ = arabidopsis.labels()
    extent = (0, arabidopsis.N_RT * RT_BIN, arabidopsis.N_MZ, 0)
    for row, treatment in enumerate(args.treatments):
        sample = int(np.flatnonzero(treatments == treatment)[0])
        image = np.asarray(features[sample]).reshape((2, arabidopsis.N_MZ, arabidopsis.N_RT))[0]
        image = np.ma.masked_less_equal(image, 0)
        ax = fig.add_subplot(grid[row, 0])
        cmap = plt.get_cmap('magma').with_extremes(bad='black')
        ax.imshow(image, aspect='auto', extent=extent, interpolation='nearest', cmap=cmap,
                  norm=LogNorm(vmin=image.min(), vmax=image.max()))
        ax.set_ylim(arabidopsis.N_MZ, MZ_SHOWN)
        ax.set_yticks([MZ_SHOWN, 500, 1000, arabidopsis.N_MZ])
        ax.set_title(f'{treatment_label(treatment)}: sample {sample + 1} (MS1)', fontsize=7)
        ax.set_ylabel('m/z (Da)')
        if row == len(args.treatments) - 1:
            ax.set_xlabel('retention time (min)')
        if row == 0:
            panel_label(ax, 'a')

    # b: test heritability per trait, mean over the folds
    for row, (treatment, path) in enumerate(zip(args.treatments, args.heritability,
                                                strict=True)):
        results = np.load(path)
        ax = fig.add_subplot(grid[row, 1])
        test = results['test']  # methods x folds x traits
        traits = np.arange(1, test.shape[2] + 1)
        methods = list(results['methods'])
        for method in ordered(methods):
            m = methods.index(method)
            ax.plot(traits, test[m].mean(axis=0), '-o', ms=2, lw=1.2, color=COLORS[method],
                    label=LABELS[method], zorder=3 if method == 'h2opt' else 2)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(traits)
        ax.set_ylabel(f'{treatment_label(treatment)}\ntest heritability')
        ax.spines[['top', 'right']].set_visible(False)
        if row == len(args.treatments) - 1:
            ax.set_xlabel('trait')
        if row == 0:
            panel_label(ax, 'b')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
                      fontsize=6, handlelength=1.2, columnspacing=0.8)

    # c: coefficients of traits 1-3 (MS1), one panel each, larger magnitudes on top
    coefficients = np.load(args.coefficients)['coefficients'][:3, 0]
    inner = grid[:, 2].subgridspec(1, 3, wspace=0.12)
    for t, color in enumerate(('tab:cyan', 'tab:pink', 'tab:olive')):
        ax = fig.add_subplot(inner[0, t], sharey=None if t == 0 else ax)
        magnitude = np.abs(coefficients[t])
        mz, rt = np.nonzero(magnitude >= SHOWN)
        order = np.argsort(magnitude[mz, rt])
        mz, rt = mz[order], rt[order]
        ax.scatter((rt + 0.5) * RT_BIN, mz, s=40 * magnitude[mz, rt], color=color, alpha=0.6,
                   linewidths=0)
        ax.set_title(f'trait {t + 1}', fontsize=7)
        ax.set_xlim(0, arabidopsis.N_RT * RT_BIN)
        ax.set_xticks([0, 10])
        ax.spines[['top', 'right']].set_visible(False)
        if t == 0:
            ax.set_ylabel(f'{treatment_label(args.treatments[0])}, MS1, m/z (Da)')
            panel_label(ax, 'c')
        else:
            ax.tick_params(labelleft=False)
        if t == 1:
            ax.set_xlabel('retention time (min)')

    # d: significant SNPs per trait
    counts = [significant_snps(path) for path in args.gwas]
    top = max(1, max(c.max() for c in counts))
    for col, (treatment, count) in enumerate(zip(args.treatments, counts, strict=True)):
        ax = fig.add_subplot(grid[:, 3 + col])
        traits = np.arange(1, len(count) + 1)
        ax.barh(traits, count, color=COLORS['h2opt'])
        for t, c in zip(traits, count, strict=True):
            ax.text(c + top * 0.03, t, str(c), va='center', fontsize=6)
        ax.set_yticks(traits)
        ax.set_ylim(traits[-1] + 0.6, 0.4)
        ax.set_xlim(0, top * 1.3)
        ax.set_title(treatment_label(treatment), fontsize=7)
        ax.set_xlabel('significant SNPs')
        if col == 0:
            ax.set_ylabel('trait')
            panel_label(ax, 'd')
        ax.spines[['top', 'right']].set_visible(False)
    save(fig, args.out)

    for treatment, path, count in zip(args.treatments, args.heritability, counts, strict=True):
        results = np.load(path)
        methods = list(results['methods'])
        means = {m: results['test'][methods.index(m)].mean() for m in ordered(methods)}
        print(treatment, 'mean test heritability over folds and traits:',
              ', '.join(f'{LABELS[m]} {v:.3f}' for m, v in means.items()),
              '| lowest H2-opt trait (fold mean):',
              f'{results["test"][methods.index("h2opt")].mean(0).min():.3f}',
              '| significant SNPs per trait:', count.tolist())


if __name__ == '__main__':
    main()
