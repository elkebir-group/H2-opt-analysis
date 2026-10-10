#!/usr/bin/env bash
# Run the H2-opt analysis pipeline (pipelines/Snakefile) in this repository's
# environment, with the flags every run needs.
#
# Usage:
#   pipelines/run.sh [--dry-run] [--cores N] [--gpu-jobs N] [-- snakemake args] -- <target>...
#
# A target is an output path relative to the repository root, such as
# data/derived/sorghum/gwas/paper_fold0.npz, or a rule name such as sorghum_all.
# With no target, the `all` rule runs.
#
# Fixed flags:
#   --rerun-triggers mtime  A code or params change never reruns (and so never
#                           deletes) existing outputs. Force a rerun with
#                           --forcerun <rule> and one target.
#   --keep-going            One failed job does not stop the independent ones.
#   --resources gpu=N       Small GPU jobs share the GPU, N at a time (default 8,
#                           which scales about linearly on one RTX 3080).
# --cores defaults to the machine's cores minus 2, for headroom. Each job's
# OpenBLAS/OpenMP/MKL threads are capped at its rule's `threads`, so the jobs
# together never use more than --cores.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cores=$(( $(nproc) > 3 ? $(nproc) - 2 : 1 ))
gpu_jobs=8
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) extra+=(--dry-run); shift ;;
        --cores) cores="$2"; shift 2 ;;
        --gpu-jobs) gpu_jobs="$2"; shift 2 ;;
        --) shift; break ;;
        *) extra+=("$1"); shift ;;
    esac
done

cd "$REPO"
# Large GPU jobs (CNNs on images) fail with "reserved but unallocated" memory under the default
# allocator, which fragments; expandable segments avoid that.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The h2opt commit each run used, for provenance (h2opt is an editable sibling
# checkout, so the lockfile does not record it).
h2opt_dir="$(uv run --no-sync python -c 'import h2opt, pathlib; print(pathlib.Path(h2opt.__file__).resolve().parent.parent)')"
echo "h2opt: $h2opt_dir @ $(git -C "$h2opt_dir" rev-parse --short HEAD)$(git -C "$h2opt_dir" diff --quiet HEAD || echo ' (uncommitted changes)')"
echo "H2-opt-analysis: $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo ' (uncommitted changes)')"

exec uv run --no-sync snakemake -s pipelines/Snakefile \
    --cores "$cores" --resources gpu="$gpu_jobs" \
    --rerun-triggers mtime --keep-going --printshellcmds --use-conda \
    "${extra[@]}" -- "$@"
