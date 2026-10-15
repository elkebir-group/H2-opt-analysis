#!/usr/bin/env bash
# Install MetaboAnalystR (not in conda) into this environment, at a pinned commit. Snakemake runs
# this script once, with the environment active, after it creates the environment.
set -euo pipefail
commit=0a9c9a7d6fb0fb654ebaa7c0bb0b5bca267e59cb
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -sSL "https://github.com/xia-lab/MetaboAnalystR/archive/${commit}.tar.gz" -o "$tmp/src.tar.gz"
tar -xzf "$tmp/src.tar.gz" -C "$tmp"
R CMD INSTALL --no-docs --no-html --no-multiarch "$tmp/MetaboAnalystR-${commit}"
