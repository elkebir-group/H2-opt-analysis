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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_t = simulation.N_TRAITS

    ev = np.load(args.evaluation)
    methods = list(ev['methods'])
    herit = np.clip(ev['herit'][:, :, 1], 0, None)  # methods x instances x traits, test plants
    values = {'truth': {t: np.clip(ev['herit_truth'][:, 1, t], 0, None) for t in range(n_t)}}
    values.update({m: {t: herit[j, :, t] for t in range(herit.shape[2])}
                   for j, m in enumerate(methods)})
    save_trait_boxplots(out / 'heritability.pdf', herit.shape[2], ['truth'] + methods, values,
                        'test heritability', width=7)

    diagonal = np.abs(ev['cor'][:, :, np.arange(n_t), np.arange(n_t)])
    values = {m: {t: diagonal[j, :, t] for t in range(n_t)} for j, m in enumerate(methods)}
    save_trait_boxplots(out / 'correlation.pdf', n_t, methods, values, '|r| with true trait')


if __name__ == '__main__':
    main()
