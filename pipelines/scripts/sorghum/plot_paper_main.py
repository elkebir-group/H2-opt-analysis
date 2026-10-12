"""Main sorghum figure of the paper (sorFig): input, heritability, coefficients, measured traits,
GWAS hits.

Panels: a, example spectra; b, mean test heritability over the folds of traits 1 to --traits, of
each method of --heritability (evaluate_heritability.py); c, wavelength coefficients of these
traits of the linear H2-opt model of one fold (--model, train_h2opt.py), for the traits as saved
(uncorrelated on the fold's training plants; the script checks them against --fold-traits); d,
median over the folds of |Pearson r| between the genotype means (over all plants) of the H2-opt
traits (--traits-files, one per fold) and the measured genotype means of --measured-traits
(Ferguson et al. 2021, Supplemental Table S12); e, Bonferroni-significant SNPs (0.05 / SNPs) of
each trait of the GWAS of one fold (--gwas, gwas.py). Writes the figure as .pdf and, beside it,
.png, and prints the numbers of each panel.

    python pipelines/scripts/sorghum/plot_paper_main.py --heritability heritability_h2opt.npz \\
        --model h2opt/fold0.pt --fold-traits h2opt/fold0.npz --fold 0 \\
        --traits-files h2opt/fold0.npz ... h2opt/fold9.npz \\
        --measured-traits SorgumTraits.tsv --gwas gwas/h2opt_fold0.npz \\
        --out figures/paper/sorFig.pdf --traits 4
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from h2opt.decorrelation import Decorrelation  # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402

from h2opt_analysis import sorghum  # noqa: E402
from h2opt_analysis.figures import COLORS  # noqa: E402
from h2opt_analysis.methods import LABELS, ordered  # noqa: E402

# Short labels of the measured traits (column names of Supplemental Table S12).
MEASURED_LABELS = {'gs': '$g_s$', 'AN': '$A_N$', 'ci/ca': '$c_i/c_a$'}


def read_measured_traits(path):
    """Genotypes, trait labels and (genotypes, traits) values (NaN where empty) of Table S12."""
    with open(path, newline='') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    header = next(r for r in rows if r and r[0] == 'Genotype')
    n = header.index('', 1)  # the trait columns end at the first empty column
    body = [r for r in rows[rows.index(header) + 1:] if r and r[0]]
    labels = []
    for name in header[1:n]:
        trait, model = name.split(' (', 1)
        labels.append(f"{MEASURED_LABELS.get(trait, trait)} ({model.replace(' model', '')}")
    values = np.array([[float(v) if v else np.nan for v in r[1:n]] for r in body])
    return np.array([r[0] for r in body]), labels, values


def genotype_means(Y, genotypes, names):
    """(names, traits) means of the rows of Y of each genotype in names."""
    return np.array([Y[genotypes == name].mean(axis=0) for name in names])


def model_coefficients(model_path, X, is_train):
    """(wavelengths, traits) coefficients of the saved traits of a linear TraitModels.

    The saved traits are the model outputs made uncorrelated on the training plants
    (h2opt.synthetic_traits), so the coefficients are the weights times Decorrelation's map."""
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    weights = np.stack([m.lin1.weight.detach().numpy()[0] for m in model.models], axis=1)
    biases = np.array([m.lin1.bias.item() for m in model.models])
    raw = X @ weights + biases
    decorrelation = Decorrelation().fit(raw[is_train])
    return weights @ decorrelation.components_, decorrelation.transform(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heritability', required=True)
    parser.add_argument('--model', required=True, help='linear H2-opt model of --fold (.pt)')
    parser.add_argument('--fold-traits', required=True, help='saved traits of --model (.npz)')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--traits-files', nargs='+', required=True, help='one .npz per fold')
    parser.add_argument('--measured-traits', required=True, help='Supplemental Table S12 (.tsv)')
    parser.add_argument('--gwas', required=True, help='GWAS of --fold (gwas.py .npz)')
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    parser.add_argument('--traits', type=int, required=True)
    args = parser.parse_args()
    k = args.traits

    X, genotypes, _ = sorghum.load()
    is_train = sorghum.folds(genotypes) != args.fold
    coefficients, traits = model_coefficients(args.model, X, is_train)
    saved = sorghum.load_traits(args.fold_traits)
    error = np.abs(traits - saved).max(axis=0) / saved.std(axis=0)
    # float32 outputs (saved) vs float64 (here)
    assert error.max() < 1e-2, f'model and saved traits differ (relative error {error})'

    fig = plt.figure(figsize=(15, 3.4))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1.1, 1.4, 1.4, 0.6], wspace=0.7)

    # a: example spectra
    ax = fig.add_subplot(grid[0])
    wavelengths = sorghum.FIRST_WAVELENGTH + np.arange(X.shape[1])
    for i in range(3):
        ax.plot(wavelengths, X[i], lw=0.8)
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('reflectance')
    ax.set_title(f'n = {len(X)} plants, {len(np.unique(genotypes))} genotypes', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    # b: mean test heritability per trait
    results = np.load(args.heritability)
    methods = list(results['methods'])
    test = results['test'][:, :, :k].mean(axis=1)  # methods x traits
    ax = fig.add_subplot(grid[1])
    order = np.arange(1, k + 1)
    for method in ordered(methods):
        m = methods.index(method)
        on_top = method == 'h2opt'  # PCH is close to H2-opt; draw H2-opt over it
        ax.plot(order, test[m], 'o-', color=COLORS[method], ms=4 if on_top else 3,
                lw=2 if on_top else 1, zorder=3 if on_top else 2, label=LABELS[method])
    ax.set_xticks(order)
    ax.set_xlabel('synthetic trait')
    ax.set_ylabel('mean test heritability')
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=6)
    ax.spines[['top', 'right']].set_visible(False)

    # c: wavelength coefficients of the traits of one fold
    inner = GridSpecFromSubplotSpec(k, 1, grid[2], hspace=0.6)
    for t in range(k):
        ax = fig.add_subplot(inner[t])
        top = np.abs(coefficients[:, t]).max()
        ax.plot(wavelengths, coefficients[:, t], lw=0.8, color=COLORS['h2opt'])
        ax.axhline(0, color='0.7', lw=0.5)
        ax.set_ylim(-1.1 * top, 1.1 * top)
        ax.set_title(f'trait {t + 1}', fontsize=7, loc='left', pad=2)
        ax.tick_params(labelsize=6)
        ax.spines[['top', 'right']].set_visible(False)
        if t < k - 1:
            ax.set_xticklabels([])
    ax.set_xlabel('wavelength (nm)')
    fig.text(grid[2].get_position(fig).x0 - 0.035, 0.5, 'coefficient', rotation=90,
             va='center', fontsize=10)

    # d: |r| of genotype means with the measured traits, median over the folds
    names, labels, measured = read_measured_traits(args.measured_traits)
    phenotyped = np.isin(names, genotypes)  # the table also has genotypes without spectra
    names, measured = names[phenotyped], measured[phenotyped]
    correlation = np.zeros((len(args.traits_files), k, measured.shape[1]))
    for f, path in enumerate(args.traits_files):
        means = genotype_means(sorghum.load_traits(path)[:, :k], genotypes, names)
        for j in range(measured.shape[1]):
            known = ~np.isnan(measured[:, j])
            for t in range(k):
                correlation[f, t, j] = np.corrcoef(means[known, t], measured[known, j])[0, 1]
    median = np.median(np.abs(correlation), axis=0).T  # measured x traits
    ax = fig.add_subplot(grid[3])
    image = ax.imshow(median, cmap='magma', vmin=0, aspect='auto')
    for (j, t), value in np.ndenumerate(median):
        ax.text(t, j, f'{value:.2f}', ha='center', va='center', fontsize=6,
                color='black' if value > 0.6 * median.max() else 'white')
    ax.set_xticks(range(k), order)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlabel('synthetic trait')
    fig.colorbar(image, ax=ax, fraction=0.08, pad=0.03, label='median |r|')

    # e: Bonferroni-significant SNPs per trait
    gwas = np.load(args.gwas)
    score = gwas['neg_log10_p']
    threshold = -np.log10(0.05 / score.shape[1])
    hits = (score > threshold).sum(axis=1)
    ax = fig.add_subplot(grid[4])
    ax.bar(np.arange(1, len(hits) + 1), hits, color=COLORS['h2opt'])
    ax.set_xticks(np.arange(1, len(hits) + 1))
    ax.set_xlabel('synthetic trait')
    ax.set_ylabel('significant SNPs')
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.spines[['top', 'right']].set_visible(False)

    for c, (label, shift) in enumerate(zip('abcde', [0.04, 0.04, 0.05, 0.06, 0.04],
                                           strict=True)):
        fig.text(grid[c].get_position(fig).x0 - shift, 0.93, label, fontsize=13,
                 fontweight='bold')
    out = Path(args.out)
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')

    print(f'b: mean test heritability, traits 1-{k}')
    for method in ordered(methods):
        print(f'  {LABELS[method]:16s}', np.round(test[methods.index(method)], 3))
    print('d: median |r| over folds (rows: measured traits; columns: synthetic traits)')
    for label, row in zip(labels, median, strict=True):
        print(f'  {label:24s}', np.round(row, 2))
    j, t = np.unravel_index(np.argmax(median), median.shape)
    print(f'  highest: trait {t + 1} with {labels[j]}, {median[j, t]:.2f}')
    print(f'e: Bonferroni threshold -log10 p {threshold:.2f} ({score.shape[1]} SNPs); '
          f'significant SNPs per trait {hits.tolist()}; '
          f'lowest p per trait {[f"{10 ** -s:.3g}" for s in score.max(axis=1)]}')


if __name__ == '__main__':
    main()
