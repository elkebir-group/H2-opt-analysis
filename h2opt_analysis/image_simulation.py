"""Image simulation: top-down plot images of plants with three heritable traits.

The true traits are those of the paper's simulation (h2opt_analysis.simulation: simplePHENOTYPES,
836 lines x 2 plants, broad-sense heritability 0.6, 0.4 and 0.2), standardized. They drive three
plant traits: leaf greenness (heritability 0.6), canopy size (0.4) and tiller number (0.2). The
images have the shape of the Miscanthus images (6 channels of 63 x 63 pixels: canopy height, then
blue, green, red, red edge and NIR reflectance) and are scaled as them (miscanthus.scale_images).

The image of a plant:
- size: canopy radius R = 12 exp(0.3 z_size + 0.05 N) pixels (N: a non-heritable stand effect).
- tillers: K ~ Poisson(60 exp(0.8 z_tillers)) tillers at random points of the canopy disk. Each
  tiller paints a Gaussian clump of width 1.5 pixels, so the canopy cover is
  c = 1 - exp(-sum of clumps): more tillers give a denser canopy with a finer texture.
- green: leaf chlorophyll u = z_green raises green, red edge and NIR and lowers blue and red
  reflectance of the leaves (4 x LEAF_GREEN per unit).
- Each reflectance band is the mixed pixel c x leaf + (1 - c) x soil. The canopy height is the
  cover times the mean height of the clumps.
Non-heritable variation of each plot: position offset (uniform, +-8 pixels), soil brightness
(x exp(0.15 N)), tiller height, pixel noise, and senesced patches. Senescence is chlorophyll loss
in patches: a mean of 3 patches per plot (Poisson with a gamma-distributed mean), each lowering u
by 8 trait standard deviations. The patches lie on the same reflectance axis as the heritable
greenness, at random places, so a linear map of the pixels cannot remove them, but a map that
reads the green pixels and ignores the senesced ones can.

The test lines are those of the spectral simulation (simulation.test_lines).
"""

import numpy as np

from h2opt_analysis import miscanthus, simulation

SIZE = 63  # pixels per side
# Reflectance of the soil and of a leaf of mean greenness: blue, green, red, red edge, NIR.
SOIL = np.array([0.08, 0.10, 0.12, 0.15, 0.18])
LEAF = np.array([0.04, 0.10, 0.05, 0.25, 0.45])
# Change of the leaf reflectance per unit of greenness.
LEAF_GREEN = np.array([-0.005, 0.01, -0.012, 0.03, 0.05])
# The design: effect sizes and non-heritable variation (see the module docstring).
SIZE_EFFECT = 0.3
TILLER_EFFECT = 0.8
GREEN_EFFECT = 4.0
STAND = 0.05
OFFSET = 8.0
SENESCENCE = 3.0
SENESCENCE_DEPTH = 8.0


def generate_images(size, tillers, green, seed=0):
    """Images (n, 6, SIZE, SIZE) float32 of plants with the given standardized trait values
    (each of length n), before scaling. seed: the non-heritable variation."""
    rng = np.random.default_rng(seed)
    n = len(size)
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    images = np.zeros((n, 6, SIZE, SIZE), dtype=np.float32)
    for i in range(n):
        radius = 12 * np.exp(SIZE_EFFECT * size[i] + STAND * rng.standard_normal())
        k = rng.poisson(60 * np.exp(TILLER_EFFECT * tillers[i]))
        center = SIZE // 2 + rng.uniform(-OFFSET, OFFSET, size=2)
        r = radius * np.sqrt(rng.uniform(size=k))
        angle = rng.uniform(0, 2 * np.pi, size=k)
        ty, tx = center[0] + r * np.sin(angle), center[1] + r * np.cos(angle)
        height = 2.0 * (1 + 0.15 * rng.standard_normal(k)) * np.sqrt(1 - (r / radius) ** 2 * 0.7)
        d2 = (yy[None] - ty[:, None, None]) ** 2 + (xx[None] - tx[:, None, None]) ** 2
        clumps = np.exp(-d2 / (2 * 1.5 ** 2))
        density = clumps.sum(axis=0)
        cover = 1 - np.exp(-density)
        canopy_height = (clumps * height[:, None, None]).sum(axis=0) / np.maximum(density, 1e-6)
        images[i, 0] = cover * canopy_height
        senesced = np.zeros((SIZE, SIZE), dtype=np.float32)
        m = rng.poisson(SENESCENCE * rng.gamma(2.0) / 2.0)
        if m:
            rr = radius * np.sqrt(rng.uniform(size=m))
            aa = rng.uniform(0, 2 * np.pi, size=m)
            py, px = center[0] + rr * np.sin(aa), center[1] + rr * np.cos(aa)
            rad = 1.5 * np.exp(0.3 * rng.standard_normal(m))
            dd = (yy[None] - py[:, None, None]) ** 2 + (xx[None] - px[:, None, None]) ** 2
            senesced = 1 - np.exp(-3 * np.exp(-dd / (2 * rad[:, None, None] ** 2)).sum(axis=0))
        u = green[i] - SENESCENCE_DEPTH * senesced  # chlorophyll of each pixel
        leaf = LEAF[:, None, None] + GREEN_EFFECT * LEAF_GREEN[:, None, None] * u[None]
        soil = SOIL * np.exp(0.15 * rng.standard_normal())
        images[i, 1:] = cover[None] * leaf + (1 - cover[None]) * soil[:, None, None]
    images[:, 1:] += 0.005 * rng.standard_normal(images[:, 1:].shape).astype(np.float32)
    images[:, 0] += 0.05 * rng.standard_normal(images[:, 0].shape).astype(np.float32)
    return images


def make_instance(instance):
    """The scaled images (n, 6, SIZE, SIZE), line names, test plants (boolean) and true traits
    (n, 3) of one instance; the true traits drive greenness, size and tillers in this order."""
    names, truth = simulation.load_truth(instance)
    tests = simulation.test_lines([simulation.load_truth(i)[0]
                                   for i in range(simulation.N_INSTANCES)])
    z = (truth - truth.mean(axis=0)) / truth.std(axis=0)
    images = generate_images(z[:, 1], z[:, 2], z[:, 0], seed=instance)
    return miscanthus.scale_images(images), names, tests[instance], truth
