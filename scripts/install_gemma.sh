#!/bin/bash
# Install GEMMA 0.97 into bin/gemma-0.97 of this repository.
# Linux: the official v0.97 release (a Guix pack, commit c760aa0), run through its bundled loader and libraries.
# macOS: the official gemma.macosx binary of the same release.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=bin/gemma-0.97
URL=https://github.com/genetics-statistics/GEMMA/releases/download/v0.97
mkdir -p "$DIR"
if [ "$(uname)" = Darwin ]; then
  curl -sSL "$URL/gemma.macosx.gz" | gunzip > "$DIR/gemma"
  chmod +x "$DIR/gemma"
else
  curl -sSL "$URL/gemma-git-gn2-0.97-c760aa0-generic-x86_64.tar.bz2" | tar xj -C "$DIR" gnu
  cat > "$DIR/gemma" <<'LAUNCHER'
#!/bin/bash
# GEMMA 0.97 (official v0.97 Linux Guix pack, commit c760aa0), run via its bundled loader.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gnu/store
exec "$D"/*-glibc-2.25/lib/ld-linux-x86-64.so.2 \
  --library-path "$(printf '%s:' "$D"/*/lib)" "$D"/*-gemma-gn2-0.97-*/bin/gemma "$@"
LAUNCHER
  chmod +x "$DIR/gemma"
fi
"$DIR/gemma" 2>&1 | head -1
