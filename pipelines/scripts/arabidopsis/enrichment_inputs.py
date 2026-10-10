"""Peak lists for MetaboAnalyst functional analysis (mummichog, web tool, default settings), one per
H2-opt trait of one Arabidopsis treatment and fold.

As the paper: the MS1 coefficient of each m/z and retention-time bin (coefficients.py) times the
standard deviation of the intensity of each precise mass in the bin (precise_masses.py); the
top-ranked precise masses by the absolute product, with their mean retention time. Writes
<out-dir>/trait<k>.txt (header "m.z" "rt", space separated, best first) for traits 1..n.

    python pipelines/scripts/arabidopsis/enrichment_inputs.py --precise precise_0d.npz \\
        --coefficients coefficients_fold0.npz --peaks 10000 --out-dir enrichment
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precise', required=True, help='precise_masses.py output')
    parser.add_argument('--coefficients', required=True, help='coefficients.py output')
    parser.add_argument('--peaks', type=int, required=True, help='masses per list')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p = np.load(args.precise)
    coefficients = np.load(args.coefficients)['coefficients'][:, 0]  # traits x m/z x rt, MS1
    for t, c in enumerate(coefficients):
        score = np.abs(c[p['mz_bin'], p['rt_bin']] * p['sd'])
        top = np.argsort(-score)[:args.peaks]
        rows = [f'{m:.4f} {r:.8f}' for m, r in zip(p['mz'][top], p['rt'][top], strict=True)]
        (out / f'trait{t + 1}.txt').write_text('"m.z" "rt"\n' + '\n'.join(rows) + '\n')
        print(f'trait {t + 1}: {len(top)} masses, m/z {p["mz"][top].min():.1f}-'
              f'{p["mz"][top].max():.1f}, top m/z {p["mz"][top[0]]:.4f}')


if __name__ == '__main__':
    main()
