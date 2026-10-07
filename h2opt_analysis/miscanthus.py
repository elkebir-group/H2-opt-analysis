"""Miscanthus multispectral drone images: 3,900 plants per population x 14 dates x 6 channels x
63 x 63 pixels, and the field design of the plants.

Two populations of half-sib families, Central and South Japan, each planted as plots of 10 plants
of one family in reps (blocks) of 130 field columns. Channel 0 is the crop surface model (height,
near zero and sometimes negative early in the season); channels 1-5 are blue, green, red, red edge
and near infrared.
"""

import types

import numpy as np

from .paths import data

POPULATIONS = ('central', 'south')
DATES = ('05/06', '05/22', '06/08', '06/18', '07/02', '07/10', '07/24', '08/08', '08/18',
         '09/04', '09/19', '10/05', '11/06', '11/28')
N_FOLDS = 5
# Ridges of PCH: as h2opt.baselines.RIDGES, the squares of noise levels, here 0.001 to 100 (RIDGES
# stop at level 10, ridge 100). With 23,814 pixels per date, PCH chose ridge 100, the largest of
# RIDGES, in 3 of the 5 South folds.
PCH_RIDGES = tuple(float(s) ** 2 for s in np.logspace(-3, 2, 21))
INPUTS = data('miscPlant', 'inputs')
_TABLES = {'central': 'Miscanthus_C.tsv', 'south': 'Miscanthus_S.tsv'}
_IMAGES = {'central': 'all.npz', 'south': 'south_all.npz'}


def plant_table(population):
    """Path of the plant table (one row per plant, in the order of the images)."""
    return INPUTS / 'data_process' / _TABLES[population]


def image_archive(population):
    """Path of the images as received: (3900, 14, 6, 63, 63) float64, compressed .npz."""
    return INPUTS / 'MSI_combine' / _IMAGES[population]


def half_sib(family):
    """True for plants of a half-sib family. Excluded: the M. x giganteus 'Illinois' clone checks,
    the 'unknown' seed lots (provenance unclear) and empty plots (blank family)."""
    family = np.char.lower(np.asarray(family).astype(str))
    return ((family != '') & (np.char.find(family, 'giganteus') < 0)
            & (np.char.find(family, 'unknown') < 0))


def design(population):
    """Field design of the plants used (half_sib).

    Returns a namespace: keep, (3900,) boolean, the plants used, to index the images with;
    families, plots, environment and fold of the kept plants. environment is (n, 2): rep and
    field column (plot number % 130), the fixed effects of the heritability (Henderson3, with
    families as groups and plots as subgroups). fold (0-4) is the paper's fold of each plant:
    np.random.seed(1), randint(5) over all 3,900 plants. The folds split plants, not families:
    the heritability removes 130 field column effects, and the plants of a few held-out families
    cover too few plots per column to estimate it."""
    table = np.loadtxt(plant_table(population), delimiter='\t', dtype=str, skiprows=1)
    family, plot, rep = np.char.strip(table[:, 7]), table[:, 2], table[:, 4]
    keep = half_sib(family)
    environment = np.column_stack([rep, plot.astype(int) % 130])
    fold = np.random.RandomState(1).randint(N_FOLDS, size=len(family))
    return types.SimpleNamespace(keep=keep, families=family[keep], plots=plot[keep],
                                 environment=environment[keep], fold=fold[keep])


def scale_images(images):
    """Each channel divided by 3 x its mean absolute value over the plants and pixels.

    images: (n, channels, height, width). The mean absolute value, not the mean, because the crop
    surface model (channel 0) is near zero and can be negative; with the factor 3, reflectance is
    roughly in [0, 1]."""
    images = np.asarray(images, dtype=np.float32)
    return images / (3 * np.abs(images).mean(axis=(0, 2, 3), keepdims=True))


def load_date(images_path, keep, date):
    """Scaled images (n, 6, 63, 63) float32 of the kept plants (keep, from design) on one date
    (index into DATES).

    images_path: the float32 (3900, 14, 6, 63, 63) .npy written by the pipeline from
    image_archive(population)."""
    images = np.load(images_path, mmap_mode='r')
    return scale_images(images[keep, date])
