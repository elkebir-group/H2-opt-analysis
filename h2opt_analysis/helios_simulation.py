"""Image simulation rendered with Helios: the plot images of image_simulation, made by a 3D plant
and radiation simulator in place of hand-made images.

The true traits, lines and test plants are those of image_simulation (the paper's three traits of
the spectral simulation, standardized). They drive the same three plant traits:
- greenness: leaf chlorophyll 40 exp(0.5 z) ug/cm2 (PROSPECT leaf optics, per plot);
- size: leaf prototype scale 0.65 exp(0.3 z + 0.05 N) (sorghum default 0.65; N: a non-heritable
  stand effect);
- tillers: shoots ~ Poisson(8 exp(0.8 z)), at least 1, uniform on a disk of radius
  0.03 sqrt(shoots) m (clump area grows with the shoot count).
Non-heritable variation of each plot: clump offset (uniform +-8 pixels), soil brightness
(x exp(0.15 N)), senescence (each leaf senescent with probability p ~ Beta(2, 10), mean 0.17,
per plot: a fraction, so that it does not track the number of shoots), and the Monte Carlo noise
of the render.

A plot is a clump of sorghum shoots of 20 days (Helios has no Miscanthus model) on soil, 3 ft
apart, seen from 50 m above in the 5 MicaSense RedEdge bands at 63 pixels per plot. The images
have the channels of the Miscanthus images: canopy height (m), then blue, green, red, red edge
and NIR, and are scaled as them (miscanthus.scale_images).

The plots are placed at random in a field and rendered in square tiles of TILE x TILE plots
(pipelines/helios/field_render), each with a ring of filler plots (random trait values, not
kept) so that every plot has neighbors.
"""

import os
import subprocess
import time

import numpy as np

TILE = 12
PX = 63  # pixels per plot side
PLOT_M = 0.9144  # 3 ft
COLUMNS = 'keep col row shoots leaf_scale chlorophyll senesced dx dy soil_scale'


def plot_inputs(z_size, z_tillers, z_green, rng):
    """Per-plot inputs (n, 7) of field_render: shoots, leaf_scale, chlorophyll, senesced, dx, dy
    and soil_scale (the columns after keep, col and row) of plots with the given standardized
    traits."""
    n = len(z_size)
    shoots = np.maximum(rng.poisson(8 * np.exp(0.8 * z_tillers)), 1)
    leaf_scale = 0.65 * np.exp(0.3 * z_size + 0.05 * rng.standard_normal(n))
    chlorophyll = 40 * np.exp(0.5 * z_green)
    senesced = rng.beta(2.0, 10.0, size=n)
    offset = rng.uniform(-8, 8, size=(n, 2)) * PLOT_M / PX
    soil_scale = np.exp(0.15 * rng.standard_normal(n))
    return np.column_stack([shoots, leaf_scale, chlorophyll, senesced, offset, soil_scale])


def tile_table(inputs, filler):
    """Rows (keep, col, row, inputs...) of one tile of TILE + 2 plots per side: the plots of inputs
    (at most TILE^2, in row-major order) inside, filler plots on the ring and on unused inside
    positions. The kept rows come in the order of inputs."""
    side = TILE + 2
    rows, f = [], 0
    for row in range(side):
        for col in range(side):
            k = (row - 1) * TILE + (col - 1)
            if 0 < row < side - 1 and 0 < col < side - 1 and k < len(inputs):
                rows.append([1, col, row, *inputs[k]])
            else:
                rows.append([0, col, row, *filler[f]])
                f += 1
    return rows


def write_table(path, rows):
    """The plots.tsv of field_render."""
    with open(path, 'w') as out:
        out.write(COLUMNS + '\n')
        for r in rows:
            out.write(f'{int(r[0]):d} {int(r[1]):d} {int(r[2]):d} {int(r[3]):d} {r[4]:.5f} '
                      f'{r[5]:.4f} {r[6]:.4f} {r[7]:.5f} {r[8]:.5f} {r[9]:.5f}\n')


def crop(tile, rows):
    """Images (kept plots, 6, PX, PX) of a tile image (6, H, W), in the order of rows. In the
    camera pixel order of field_render, x runs along -col and y along -row: plot (col, row) is
    the block at x0 = (side - 1 - col) PX, y0 = (side - 1 - row) PX (checked by rendering marked
    plots)."""
    side = tile.shape[1] // PX
    return np.stack([tile[:, (side - 1 - r) * PX:(side - r) * PX,
                          (side - 1 - c) * PX:(side - c) * PX]
                     for keep, c, r in ((int(t[0]), int(t[1]), int(t[2])) for t in rows) if keep])


def render(z_size, z_tillers, z_green, seed, binary, workdir):
    """Unscaled images (n, 6, PX, PX) float32 of plots with the given standardized traits.

    binary: the field_render executable (run in its build directory); workdir: the directory of
    the per-tile tables and images. seed: the non-heritable variation and the field layout."""
    rng = np.random.default_rng(seed)
    n = len(z_size)
    inputs = plot_inputs(z_size, z_tillers, z_green, rng)
    order = rng.permutation(n)  # plots in field order
    os.makedirs(workdir, exist_ok=True)
    binary = os.path.abspath(binary)
    side = TILE + 2
    images = np.zeros((n, 6, PX, PX), dtype=np.float32)
    for t, start in enumerate(range(0, n, TILE * TILE)):
        plots = order[start:start + TILE * TILE]
        z = rng.standard_normal((side * side, 3))
        rows = tile_table(inputs[plots], plot_inputs(z[:, 0], z[:, 1], z[:, 2], rng))
        table = os.path.abspath(os.path.join(workdir, f'tile{t}.tsv'))
        out = os.path.abspath(os.path.join(workdir, f'tile{t}.bin'))
        write_table(table, rows)
        t0 = time.time()
        log = subprocess.run([binary, table, out, str(seed * 1000 + t)],
                             cwd=os.path.dirname(binary), capture_output=True, text=True,
                             check=True).stdout
        tile = np.fromfile(out, dtype=np.float32).reshape(6, side * PX, side * PX)
        images[plots] = crop(tile, rows)
        os.remove(out)  # 19 MB per tile
        print(f'tile {t}: {len(plots)} plots, {time.time() - t0:.1f} s; '
              f'{log.strip().splitlines()[-1]}', flush=True)
    return images
