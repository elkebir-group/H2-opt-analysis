"""Heritability of each method's traits on one Arabidopsis treatment (boxplots over the folds).

One figure of the test heritability per trait (as the paper's Fig. arabiResult b) and one of the
train heritability. Reads the .npz of evaluate_heritability.py; method labels and colors come from
the slugs saved with the values.

    python pipelines/scripts/arabidopsis/plot_heritability.py --heritability heritability.npz \\
        --out-test heritability_test.pdf --out-train heritability_train.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import numpy as np  # noqa: E402

from h2opt_analysis.figures import save_trait_boxplots  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heritability', required=True)
    parser.add_argument('--out-test', required=True)
    parser.add_argument('--out-train', required=True)
    args = parser.parse_args()

    results = np.load(args.heritability)
    methods = list(results['methods'])
    for split, path in (('test', args.out_test), ('train', args.out_train)):
        values = results[split]  # methods x folds x traits
        boxes = {m: {t: values[i, :, t] for t in range(values.shape[2])}
                 for i, m in enumerate(methods)}
        save_trait_boxplots(path, values.shape[2], methods, boxes,
                            f'{split} heritability (ANOVA)', width=7)


if __name__ == '__main__':
    main()
