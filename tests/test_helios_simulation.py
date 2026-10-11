"""Tiles of the Helios image simulation: the plots go in and come out in the same order."""

import numpy as np

from h2opt_analysis import helios_simulation as hs


def test_tile_table_and_crop_keep_the_plot_order():
    rng = np.random.default_rng(0)
    n = hs.TILE * hs.TILE - 5  # a last, partly filled tile
    inputs = hs.plot_inputs(*rng.standard_normal((3, n)), rng)
    side = hs.TILE + 2
    filler = hs.plot_inputs(*rng.standard_normal((3, side * side)), rng)
    rows = hs.tile_table(inputs, filler)
    assert len(rows) == side * side
    kept = np.array([r[3:] for r in rows if r[0] == 1])
    np.testing.assert_array_equal(kept, inputs)
    # a tile image in field_render's pixel order, each plot block filled with its row index
    tile = np.zeros((6, side * hs.PX, side * hs.PX), dtype=np.float32)
    for k, (_keep, col, row) in enumerate(r[:3] for r in rows):
        y0, x0 = (side - 1 - row) * hs.PX, (side - 1 - col) * hs.PX
        tile[:, y0:y0 + hs.PX, x0:x0 + hs.PX] = k
    images = hs.crop(tile, rows)
    kept_rows = [k for k, r in enumerate(rows) if r[0] == 1]
    assert images.shape == (n, 6, hs.PX, hs.PX)
    np.testing.assert_array_equal(images[:, 0, 0, 0], kept_rows)
    assert all((im == im.flat[0]).all() for im in images)
