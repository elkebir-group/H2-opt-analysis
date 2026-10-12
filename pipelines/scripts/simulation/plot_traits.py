"""Trait figures of the simulation (paper Fig. 3a-b).

Writes into --out-dir: heritability.pdf (test heritability of synthetic traits 1-5 and of the true
traits) and correlation.pdf (|r| of true trait t with synthetic trait t).

    python pipelines/scripts/simulation/plot_traits.py --evaluation evaluation.npz \\
        --out-dir figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import numpy as np  # noqa: E402

from h2opt_analysis import simulation  # noqa: E402
from h2opt_analysis.figures import save_trait_boxplots  # noqa: E402
from h2opt_analysis.methods import ordered  # noqa: E402


def heritability_values(ev):
    """Test heritability (clipped at 0) of the true traits and of each method's traits.

    ev: an evaluate.py result. Returns the slugs ('truth' first, then the methods in figure order),
    the values (values[slug][t] is an array over the instances) and the number of synthetic
    traits."""
    methods = list(ev['methods'])
    herit = np.clip(ev['herit'][:, :, 1], 0, None)  # methods x instances x traits, test plants
    values = {'truth': {t: np.clip(ev['herit_truth'][:, 1, t], 0, None)
                        for t in range(simulation.N_TRAITS)}}
    values.update({m: {t: herit[j, :, t] for t in range(herit.shape[2])}
                   for j, m in enumerate(methods)})
    return ['truth'] + ordered(methods), values, herit.shape[2]


def correlation_values(ev):
    """|r| of true trait t with synthetic trait t of each method: the slugs (in figure order) and
    the values (values[slug][t] is an array over the instances)."""
    methods = list(ev['methods'])
    n_t = simulation.N_TRAITS
    diagonal = np.abs(ev['cor'][:, :, np.arange(n_t), np.arange(n_t)])
    return ordered(methods), {m: {t: diagonal[j, :, t] for t in range(n_t)}
                              for j, m in enumerate(methods)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ev = np.load(args.evaluation)
    slugs, values, n_traits = heritability_values(ev)
    save_trait_boxplots(out / 'heritability.pdf', n_traits, slugs, values, 'test heritability',
                        width=7)
    slugs, values = correlation_values(ev)
    save_trait_boxplots(out / 'correlation.pdf', simulation.N_TRAITS, slugs, values,
                        '|r| with true trait')


if __name__ == '__main__':
    main()
