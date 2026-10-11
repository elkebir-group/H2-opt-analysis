#!/usr/bin/env bash
# Clone Helios at one commit and build pipelines/helios/field_render against it.
# Needs git, cmake, a C++17 compiler and the Vulkan loader (the radiation plugin falls back from
# OptiX to Vulkan compute when there is no CUDA toolkit).
#
#   build_helios.sh <repository> <commit> <field_render source dir> <out dir> <jobs>
# Writes <out dir>/Helios (the checkout) and <out dir>/build/field_render.
set -euo pipefail
repository=$1 commit=$2 source=$(realpath "$3") out=$(realpath -m "$4") jobs=$5

mkdir -p "$out"
if [ ! -d "$out/Helios/.git" ]; then
    git clone --quiet "$repository" "$out/Helios"
fi
git -C "$out/Helios" fetch --quiet origin
git -C "$out/Helios" checkout --quiet "$commit"
echo "Helios $(git -C "$out/Helios" log --oneline -1)"

vulkan=$(ldconfig -p | awk '/libvulkan\.so\.1 / {print $NF; exit}')
mkdir -p "$out/build"
cmake -S "$source" -B "$out/build" -DCMAKE_BUILD_TYPE=Release -DHELIOS_DIR="$out/Helios" \
    -DVulkan_LIBRARIES="$vulkan"
cmake --build "$out/build" -j "$jobs"
