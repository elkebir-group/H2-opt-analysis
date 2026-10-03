"""Run a command over a grid of parameters, several jobs at once, each with a capped number of CPU threads.

    uv run python -m h2opt_analysis.launch --jobs 8 --threads 1 --grid fold=0-9 seed=0,1 \\
        -- python scripts/sorghum/train_h2opt.py --fold {fold} --seed {seed} --device cuda

Every combination of the grid values fills the {name} fields of the command. Output of each job goes to
<log-dir>/<name>_<value>_....log. Small GPU jobs are limited by Python overhead, not by the GPU, so up to
~8 concurrent jobs on one GPU scale about linearly. Keep jobs x threads within the machine's cores.
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_values(text):
    """'0-9' -> 0..9, '0,2,5' -> those values, anything else -> that single value."""
    values = []
    for part in text.split(','):
        if '-' in part and all(p.isdigit() for p in part.split('-', 1)):
            low, high = map(int, part.split('-', 1))
            values.extend(str(v) for v in range(low, high + 1))
        else:
            values.append(part)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jobs', type=int, default=4, help='jobs running at once')
    parser.add_argument('--threads', type=int, default=1, help='CPU threads per job (OpenBLAS/OpenMP/MKL/torch)')
    parser.add_argument('--grid', nargs='+', default=[], metavar='NAME=VALUES', help='e.g. fold=0-9 seed=0,1')
    parser.add_argument('--log-dir', default='results/logs')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='-- command with {name} fields')
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('no command given')
    names = [g.split('=', 1)[0] for g in args.grid]
    combos = [dict(zip(names, values)) for values in itertools.product(*(parse_values(g.split('=', 1)[1]) for g in args.grid))]
    cores = os.cpu_count() or 1
    if args.jobs * args.threads > cores:
        print(f'warning: {args.jobs} jobs x {args.threads} threads exceeds {cores} cores', file=sys.stderr)

    env = dict(os.environ, PYTHONUNBUFFERED='1')
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[var] = str(args.threads)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pending, running, failed = list(combos), [], []
    start = time.time()
    while pending or running:
        while pending and len(running) < args.jobs:
            combo = pending.pop(0)
            argv = [part.format(**combo) for part in command]
            log = log_dir / (('_'.join(f'{k}{v}' for k, v in combo.items()) or 'job') + '.log')
            with open(log, 'w') as out:
                process = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env)
            running.append((process, combo, log))
            print(f'started {combo} -> {log}', flush=True)
        time.sleep(1)
        for job in list(running):
            process, combo, log = job
            if process.poll() is not None:
                running.remove(job)
                status = 'done' if process.returncode == 0 else f'FAILED (exit {process.returncode})'
                if process.returncode != 0:
                    failed.append(combo)
                print(f'{status} {combo} [{time.time() - start:.0f}s]', flush=True)

    print(f'{len(combos) - len(failed)}/{len(combos)} jobs succeeded')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
