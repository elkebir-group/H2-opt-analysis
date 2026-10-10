"""The Miscanthus GWAS figure of the paper's supplement (miscGWAS): Manhattan and QQ plots of trait
1 of hybrid H2-opt in the first fold, for each date with a significant SNP.

A SNP is significant when p < 0.05 / SNPs (Bonferroni). Each population is one page of the .pdf
(figures.plot_gwas, one row per date); the .png next to it holds the pages one above the other.
A population without a significant SNP gets no page. The function significant_counts also makes
panel e of miscFig (plot_paper_main.py).

    python pipelines/scripts/miscanthus/plot_paper_gwas.py \\
        --gwas central/gwas/date0.npz ... central/gwas/date13.npz \\
        south/gwas/date0.npz ... south/gwas/date13.npz --out figures/paper/miscGWAS.pdf
"""

import argparse
import io

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from h2opt_analysis import miscanthus  # noqa: E402
from h2opt_analysis.figures import plot_gwas  # noqa: E402

FOLD = 0  # the paper's "first cross-validation split"


def significant_counts(files, fold=FOLD):
    """Number of Bonferroni-significant SNPs of one fold, per file (one file per date)."""
    counts = []
    for file in files:
        with np.load(file) as f:
            score = f['neg_log10_p'][fold]
        counts.append(int((score > -np.log10(0.05 / len(score))).sum()))
    return np.array(counts)


def date_summary(files, dates):
    """A gwas.summarize-like dict with one row per date (index into files), fold FOLD."""
    rows = [np.load(files[d]) for d in dates]
    for f in rows:
        assert (f['chr'] == rows[0]['chr']).all() and (f['ps'] == rows[0]['ps']).all()
    return {'chr': rows[0]['chr'], 'ps': rows[0]['ps'],
            'neg_log10_p': np.stack([f['neg_log10_p'][FOLD] for f in rows]),
            'lambda_gc': np.array([f['lambda_gc'][FOLD] for f in rows])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gwas', nargs='+', required=True,
                        help='one file per date, Central then South')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    n_dates = len(miscanthus.DATES)
    assert len(args.gwas) == 2 * n_dates, 'give one file per date and population'

    pngs = []
    with PdfPages(args.out) as pdf:
        for p, population in enumerate(miscanthus.POPULATIONS):
            files = args.gwas[p * n_dates:(p + 1) * n_dates]
            counts = significant_counts(files)
            print(population, 'significant SNPs per date, fold 0:', counts.tolist())
            dates = np.flatnonzero(counts)
            if not len(dates):
                continue
            summary = date_summary(files, dates)
            labels = [f'{miscanthus.DATES[d]}\n({counts[d]} SNPs)' for d in dates]
            title = f'{population.capitalize()}, H2-opt trait 1, first fold'
            # plot_gwas saves without a format: the rcParam sets the format of each target.
            plt.rcParams['savefig.format'] = 'pdf'
            plot_gwas(pdf, summary, range(len(dates)), labels, title)
            plt.rcParams['savefig.format'] = 'png'
            buffer = io.BytesIO()
            plot_gwas(buffer, summary, range(len(dates)), labels, title, dpi=150)
            pngs.append(buffer)
        if not pngs:
            fig = plt.figure(figsize=(4, 1))
            fig.text(0.5, 0.5, 'no significant SNP', ha='center')
            pdf.savefig(fig)
            plt.close(fig)
    png = str(args.out).removesuffix('.pdf') + '.png'
    if pngs:
        images = [plt.imread(io.BytesIO(b.getvalue()), format='png') for b in pngs]
        width = max(i.shape[1] for i in images)
        images = [np.pad(i, ((0, 0), (0, width - i.shape[1]), (0, 0)), constant_values=1)
                  for i in images]
        plt.imsave(png, np.concatenate(images))
    else:
        plt.imsave(png, np.ones((10, 10, 3)))


if __name__ == '__main__':
    main()
