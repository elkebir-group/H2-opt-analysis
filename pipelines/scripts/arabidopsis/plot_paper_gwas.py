"""Manhattan and QQ plots of the H2-opt traits of one Arabidopsis treatment that have significant
SNPs (images/metab0d_GWAS, metab6d_GWAS).

Reads a gwas.summarize .npz (one fold) and plots, with h2opt_analysis.figures.plot_gwas, every
trait with at least one SNP above the Bonferroni threshold 0.05 / SNPs. The counts come from the
file. When no trait has a significant SNP, the figure says so. Saves --out (.pdf) and a .png next
to it.

    python pipelines/scripts/arabidopsis/plot_paper_gwas.py --gwas 0d/gwas/fold0.npz \\
        --title '0 day, fold 0' --out metab0d_GWAS.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis.figures import plot_gwas  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gwas', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    summary = dict(np.load(args.gwas))
    score = summary['neg_log10_p']
    counts = (score > -np.log10(0.05 / score.shape[1])).sum(axis=1)
    rows = np.flatnonzero(counts)
    print(f'{score.shape[1]} SNPs; significant SNPs per trait: {counts.tolist()}')
    title = f'{args.title}: traits with SNPs above 0.05 / {score.shape[1]:,} SNPs'
    png = args.out.removesuffix('.pdf') + '.png'
    if len(rows) == 0:
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.text(0.5, 0.5, 'no trait has a significant SNP', ha='center', va='center')
        ax.set_axis_off()
        fig.suptitle(title, fontsize=9)
        for path in (args.out, png):
            fig.savefig(path)
        plt.close(fig)
        return
    labels = [f'trait {r + 1} ({counts[r]} SNPs)' for r in rows]
    plot_gwas(args.out, summary, rows, labels, title)
    plot_gwas(png, summary, rows, labels, title)


if __name__ == '__main__':
    main()
