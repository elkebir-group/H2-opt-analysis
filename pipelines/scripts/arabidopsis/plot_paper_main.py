"""Main Arabidopsis figure of the paper (Fig. arabiResult, images/metabFig), panels a to e, one row
per treatment, with the treatment at the right of the row (as in simFig).

a: the binned MS1 input matrices of the first three samples of each treatment as a stack (log
scale), with the number of samples n and of accessions c. b: test heritability per trait of H2-opt
and every baseline, mean over the folds. c: coefficients of traits 1-3 of each treatment's model
(MS1 only), the ones with magnitude at least 2% of the trait's largest, the marker area
proportional to the magnitude relative to the largest. d: SNPs above the Bonferroni threshold
0.05 / SNPs, per trait. e: pathway enrichment of each trait, the adjusted Fisher p-value
(AdjP.Fisher of pathway_enrichment.R, adjusted over the pathways of one peak list) on a log color
scale, every pathway with p < 0.05 for a trait of either treatment; an asterisk marks p < 0.05.
Not the mummichog p-value P(Gamma): it is below 0.05 for most pathways also for a peak list ranked
without a trait (by intensity SD alone), while the adjusted Fisher p-value of that list is at
least 0.5.

Drawn at the text width of the paper (6.5 in) in the style of simFig. Saves --out (.pdf) and a
.png next to it.

    python pipelines/scripts/arabidopsis/plot_paper_main.py --features features.npy \\
        --treatments 0d 6d --heritability 0d/heritability.npz 6d/heritability.npz \\
        --coefficients {0d,6d}/h2opt/coefficients_fold0.npz \\
        --gwas 0d/gwas/fold0.npz 6d/gwas/fold0.npz \\
        --pathways {0d,6d}/enrichment/trait{1..10}_pathways.csv \\
        --out metabFig.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from h2opt_analysis import arabidopsis  # noqa: E402
from h2opt_analysis.figures import (  # noqa: E402
    COLORS,
    LETTER_SIZE,
    MATH_RC,
    SIM_FIG_RC,
    TICK_SIZE,
)
from h2opt_analysis.methods import LABELS, ordered  # noqa: E402

# Minutes per retention-time bin.
RT_BIN = 0.5
# Smallest coefficient magnitude shown, as a fraction of the trait's largest.
SHOWN = 0.02
# Smallest m/z shown in panels a and c (all bins below it are empty).
MZ_SHOWN = 100
STACK = 3  # samples in the stack of panel a
P_COLUMN = 'AdjP.Fisher'
ALPHA = 0.05
# KEGG pathway names, shortened for panel e.
SHORT = {'Alanine, aspartate and glutamate metabolism': 'Ala, Asp and Glu',
         'Citrate cycle (TCA cycle)': 'TCA cycle',
         'Glyoxylate and dicarboxylate metabolism': 'glyoxylate'}
TRAIT_STYLE = (('tab:cyan', 's'), ('tab:pink', 'X'), ('tab:olive', 'P'))
FIG_SIZE = (6.5, 2.35)
# Rows of panels in inches: the bottoms of the first and the second treatment, and their height.
ROW_BOTTOM = (1.27, 0.33)
ROW_HEIGHT = 0.6
# Row labels (the treatments) at the right of the rows, as in simFig.
ROW_LABEL_X = 6.33
ROW_LABEL_SIZE = 7.5


def treatment_label(treatment):
    """'0d' -> '0 day'."""
    return f'{treatment.removesuffix("d")} day'


def short_name(pathway):
    """Pathway name for panel e: SHORT, else the KEGG name without ' metabolism' and
    ' biosynthesis', in lower case."""
    if pathway in SHORT:
        return SHORT[pathway]
    name = pathway.removesuffix(' metabolism').removesuffix(' biosynthesis')
    return name[0].lower() + name[1:]


def save(fig, path):
    """Save path (.pdf) and a .png next to it."""
    fig.savefig(path)
    fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=300)
    plt.close(fig)


def significant_snps(path):
    """SNPs per trait above the Bonferroni threshold 0.05 / SNPs of a gwas.summarize .npz."""
    score = np.load(path)['neg_log10_p']
    return (score > -np.log10(0.05 / score.shape[1])).sum(axis=1)


def load_pathways(tables, n_treatments):
    """P_COLUMN p-values (treatments x traits x pathways, 1 for a pathway without hits in a list)
    and the pathway names."""
    per_treatment = np.array_split(np.array(tables), n_treatments)
    frames = [[pd.read_csv(path, index_col=0)[P_COLUMN] for path in paths]
              for paths in per_treatment]
    pathways = sorted(set().union(*(f.index for row in frames for f in row)))
    p = np.array([[f.reindex(pathways).fillna(1).to_numpy() for f in row] for row in frames])
    return p, pathways


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--treatments', nargs='+', required=True)
    parser.add_argument('--heritability', nargs='+', required=True, help='one per treatment')
    parser.add_argument('--coefficients', nargs='+', required=True,
                        help='coefficients.py output, one per treatment')
    parser.add_argument('--gwas', nargs='+', required=True, help='one per treatment')
    parser.add_argument('--pathways', nargs='+', required=True,
                        help='pathway tables, traits 1 to k of each treatment in order')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if not (len(args.treatments) == len(args.heritability) == len(args.coefficients)
            == len(args.gwas) == 2):
        parser.error('give two treatments, and one heritability, coefficients and GWAS file per '
                     'treatment')
    if len(args.pathways) % len(args.treatments):
        parser.error('give the same number of pathway tables for each treatment')

    features = np.load(args.features, mmap_mode='r')
    _, sample_treatments, _ = arabidopsis.labels()
    counts = [significant_snps(path) for path in args.gwas]
    p, pathways = load_pathways(args.pathways, len(args.treatments))
    keep = [i for i in range(len(pathways)) if p[:, :, i].min() < ALPHA]
    p, pathways = p[:, :, keep], [pathways[i] for i in keep]

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig = plt.figure(figsize=FIG_SIZE)

        def letter(x, text):
            """Panel letter x inches from the left, above the first row."""
            fig.text(x / FIG_SIZE[0], (ROW_BOTTOM[0] + ROW_HEIGHT + 0.08) / FIG_SIZE[1], text,
                     fontsize=LETTER_SIZE, fontweight='bold', va='bottom')

        def axes(x, row, width, rows=1, height=None, dy=0.0):
            """Axes at x inches from the left, width inches wide, in row 0 (first treatment),
            row 1 or, with rows=2, spanning both."""
            bottom = ROW_BOTTOM[1] if rows == 2 else ROW_BOTTOM[row]
            if height is None:
                height = ROW_HEIGHT + (ROW_BOTTOM[0] - ROW_BOTTOM[1]) * (rows - 1)
            return fig.add_axes((x / FIG_SIZE[0], (bottom + dy) / FIG_SIZE[1],
                                 width / FIG_SIZE[0], height / FIG_SIZE[1]))

        for row, treatment in enumerate(args.treatments):
            last = row == len(args.treatments) - 1

            # a: a stack of input matrices, the first sample in front
            samples = np.flatnonzero(sample_treatments == treatment)
            _, accessions, _ = arabidopsis.design(treatment)
            size, step = 0.46, 0.06
            for k in reversed(range(STACK)):
                image = np.asarray(features[samples[k]]).reshape(
                    (2, arabidopsis.N_MZ, arabidopsis.N_RT))[0, MZ_SHOWN:]
                image = np.ma.masked_less_equal(image, 0)
                ax = axes(0.42 + step * k, row, size, height=size, dy=step * k)
                ax.imshow(image, aspect='auto', interpolation='none',
                          cmap=plt.get_cmap('magma').with_extremes(bad='black'),
                          norm=LogNorm(vmin=image.min(), vmax=image.max()),
                          extent=(0, arabidopsis.N_RT * RT_BIN, arabidopsis.N_MZ, MZ_SHOWN))
                for spine in ax.spines.values():
                    spine.set(linewidth=0.6, edgecolor='white' if k else 'black')
                if k:
                    ax.set_xticks([])
                    ax.set_yticks([])
            ax.set_yticks([MZ_SHOWN, 500, 1000, arabidopsis.N_MZ])
            ax.set_xticks([0, 10, 20])
            ax.set_ylabel('$m/z$ (Da)')
            if last:
                ax.set_xlabel('retention time (min)')
            fig.text(0.42 / FIG_SIZE[0],
                     (ROW_BOTTOM[row] + size + step * (STACK - 1) + 0.04) / FIG_SIZE[1],
                     f'$n = {len(accessions)}$, $c = {len(np.unique(accessions))}$',
                     fontsize=TICK_SIZE)

            # b: test heritability per trait, mean over the folds
            results = np.load(args.heritability[row])
            test = results['test']  # methods x folds x traits
            traits = np.arange(1, test.shape[2] + 1)
            methods = list(results['methods'])
            ax = axes(1.62, row, 1.1)
            for method in reversed(ordered(methods)):
                ax.plot(traits, test[methods.index(method)].mean(axis=0), '-o', ms=1.8,
                        color=COLORS[method], label=LABELS[method])
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xticks(traits, traits if last else [])
            ax.set_ylabel('test heritability')
            ax.spines[['top', 'right']].set_visible(False)
            if last:
                ax.set_xlabel('synthetic trait')
            else:
                # above the axes, in the place of the titles of the other panels
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], loc='lower left',
                          bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
                          fontsize=TICK_SIZE - 0.5, handlelength=1.0, columnspacing=0.5,
                          handletextpad=0.25, labelspacing=0.1, borderaxespad=0.1,
                          borderpad=0)

            # c: coefficients of traits 1-3 (MS1), larger magnitudes on top
            coefficients = np.load(args.coefficients[row])['coefficients'][:3, 0]
            ax = axes(3.2, row, 0.6)
            for t, (color, marker) in enumerate(TRAIT_STYLE):
                magnitude = np.abs(coefficients[t]) / np.abs(coefficients[t]).max()
                mz, rt = np.nonzero(magnitude >= SHOWN)
                order = np.argsort(magnitude[mz, rt])
                mz, rt = mz[order], rt[order]
                ax.scatter((rt + 0.5) * RT_BIN, mz, s=6 * magnitude[mz, rt], color=color,
                           marker=marker, alpha=0.6, linewidths=0, label=f'trait {t + 1}')
            ax.set_xlim(0, arabidopsis.N_RT * RT_BIN)
            ax.set_ylim(MZ_SHOWN, 1300)
            ax.set_yticks([MZ_SHOWN, 500, 1000])
            ax.set_xticks([0, 10, 20], [0, 10, 20] if last else [])
            ax.set_ylabel('$m/z$ (Da)')
            ax.spines[['top', 'right']].set_visible(False)
            if last:
                ax.set_xlabel('retention time (min)')
            else:
                legend = ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2,
                                   frameon=False, fontsize=TICK_SIZE - 0.5, handletextpad=0,
                                   columnspacing=0.4, labelspacing=0.1, borderaxespad=0.1,
                                   borderpad=0)
                for handle in legend.legend_handles:
                    handle.set_sizes([8])
                    handle.set_alpha(1)

            # d: significant SNPs per trait
            ax = axes(4.27, row, 0.46)
            ax.bar(traits, counts[row], color=COLORS['h2opt'], width=0.7)
            ax.set_xticks(traits, [t if last and t in (1, 5, 10) else '' for t in traits])
            ax.set_ylim(0, max(c.max() for c in counts) * 1.05)
            ax.set_ylabel('significant SNPs')
            ax.spines[['top', 'right']].set_visible(False)
            if last:
                ax.set_xlabel('synthetic trait')

            # e: pathway enrichment
            ax = axes(5.29, row, 0.65)
            norm = LogNorm(vmin=1e-3, vmax=1)
            values = np.clip(p[row].T, 1e-3, 1)  # pathways x traits
            image = ax.pcolormesh(np.arange(values.shape[1] + 1) - 0.5,
                                  np.arange(values.shape[0] + 1) - 0.5, values, cmap='magma',
                                  norm=norm)
            ax.set_ylim(values.shape[0] - 0.5, -0.5)
            r, c = np.nonzero(p[row].T < ALPHA)
            ax.scatter(c, r + 0.25, marker='*', s=5, c='white', linewidths=0)
            ax.set_xticks(range(len(traits)),
                          [t if last and t in (1, 5, 10) else '' for t in traits])
            ax.set_yticks(range(len(pathways)), [short_name(n) for n in pathways],
                          fontsize=TICK_SIZE - 1.5, linespacing=0.9)
            ax.tick_params(length=0)
            if last:
                ax.set_xlabel('synthetic trait')
            bar = fig.colorbar(image, cax=axes(5.99, row, 0.05), ticks=[1e-3, 1e-2, ALPHA, 1])
            bar.ax.set_yticklabels(['$10^{-3}$', '$10^{-2}$', '$0.05$', '$1$'])
            bar.ax.tick_params(labelsize=TICK_SIZE - 1, length=1.5)
            bar.outline.set_linewidth(0.6)
            fig.text(ROW_LABEL_X / FIG_SIZE[0], (ROW_BOTTOM[row] + ROW_HEIGHT / 2) / FIG_SIZE[1],
                     treatment_label(treatment), rotation=-90, ha='left', va='center',
                     fontsize=ROW_LABEL_SIZE, fontweight='bold')

        for x, text in ((0.0, 'a'), (1.3, 'b'), (2.92, 'c'), (4.02, 'd'), (4.75, 'e')):
            letter(x, text)
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
    print(f'e: {len(pathways)} pathways: ' + '; '.join(
        f'{name} (min p {p[:, :, i].min():.2g})' for i, name in enumerate(pathways)))


if __name__ == '__main__':
    main()
