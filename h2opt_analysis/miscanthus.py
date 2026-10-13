"""Miscanthus multispectral drone images: 3,900 plants per population x 14 dates x 6 channels x
63 x 63 pixels, and the field design of the plants.

Two populations of half-sib families, Central and South Japan, each planted as plots of 10 plants
of one family in reps (blocks) of 130 field columns. Channel 0 is the crop surface model (height,
near zero and sometimes negative early in the season); channels 1-5 are blue, green, red, red edge
and near infrared.
"""

import types

import numpy as np
from h2opt import spatial

from .paths import data

POPULATIONS = ('central', 'south')
DATES = ('05/06', '05/22', '06/08', '06/18', '07/02', '07/10', '07/24', '08/08', '08/18',
         '09/04', '09/19', '10/05', '11/06', '11/28')
N_FOLDS = 5
INPUTS = data('miscPlant', 'inputs')
_TABLES = {'central': 'Miscanthus_C.tsv', 'south': 'Miscanthus_S.tsv'}
_IMAGES = {'central': 'all.npz', 'south': 'south_all.npz'}
# PLINK sets of the genotyped plants. The .fam beside each .bed was overwritten with a subset by
# the original GWAS code; the full sample list of the .bed is the _copy.fam (GENOTYPE_SAMPLES).
GENOTYPES = {'central': data('miscPlant', 'SNP', 'bed', 'MsiC_annotated'),
             'south': data('miscPlant', 'SNP', 'bed', 'MsiS_annotated')}
GENOTYPE_SAMPLES = {p: GENOTYPES[p].with_name(GENOTYPES[p].name + '_copy.fam')
                    for p in POPULATIONS}


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
    np.random.seed(1), randint(5) over all 3,900 plants."""
    table = np.loadtxt(plant_table(population), delimiter='\t', dtype=str, skiprows=1)
    family, plot, rep = np.char.strip(table[:, 7]), table[:, 2], table[:, 4]
    keep = half_sib(family)
    environment = np.column_stack([rep, plot.astype(int) % 130])
    fold = np.random.RandomState(1).randint(N_FOLDS, size=len(family))
    return types.SimpleNamespace(keep=keep, families=family[keep], plots=plot[keep],
                                 environment=environment[keep], fold=fold[keep])


def neighbor_weights(design, width):
    """Papadakis neighbor matrix of the kept plants (design): row i averages the plants of the
    plots within width field columns of plant i's plot in the same rep, without plant i's own
    plot and without other plots of its family. Rows sum to 1 (n x n, float32)."""
    rep, column = design.environment[:, 0], design.environment[:, 1].astype(int)
    near = ((rep[:, None] == rep[None]) & (np.abs(column[:, None] - column[None]) <= width)
            & (design.plots[:, None] != design.plots[None])
            & (design.families[:, None] != design.families[None])).astype(np.float32)
    return near / near.sum(axis=1, keepdims=True)


def papadakis(y, weights, rows):
    """Nearest-neighbor adjustment (Papadakis 1937) of the trait y (all kept plants, NaN where
    missing), with the slope fitted on the plants rows: h2opt.spatial.papadakis."""
    return spatial.papadakis(y, weights, rows)


def genotype_rows(population, fam):
    """Row in the .fam file fam of each kept plant (design order), -1 for a plant without
    genotypes. A sample ID is <plate well>_<population>_<plot>_<plant>, e.g. B2FP10D_MsiC_213_5.

    A plant with two samples (9 in Central, 1 in South) gets -1: the two samples of each such
    plant agree at 77-92% of SNPs, no more than two random plants (85%), so the label of at least
    one of them is wrong and neither can be trusted."""
    table = np.loadtxt(plant_table(population), delimiter='\t', dtype=str, skiprows=1)
    plants = np.char.add(np.char.add(np.char.strip(table[:, 2]), '_'), np.char.strip(table[:, 3]))
    plants = plants[design(population).keep]
    samples = np.loadtxt(fam, dtype=str, usecols=0, ndmin=1)
    rows = {}
    for r, sample in enumerate(samples):
        rows.setdefault('_'.join(sample.split('_')[2:]), []).append(r)
    return np.array([rows[p][0] if len(rows.get(p, [])) == 1 else -1 for p in plants])


def clip_artifacts(images):
    """Each channel clipped to [q1 - 2 (q99 - q1), q99 + 2 (q99 - q1)], with q1 and q99 its 1% and
    99% quantiles over the plants and pixels.

    images: (n, channels, height, width). The crop surface model has reconstruction errors:
    patches of heights of 10 m or more, or down to -50 m (e.g. South 06/18, 07/10, 10/05). After
    scaling they are 10 to 20 times the largest regular value, and the CNN extrapolates on them.
    The fence keeps every reflectance value but one (South 10/05) and clips 9,222 crop surface
    model pixels in Central and 24,987 in South, of all dates."""
    images = np.asarray(images, dtype=np.float32)
    low, high = np.quantile(images, [0.01, 0.99], axis=(0, 2, 3), keepdims=True)
    return np.clip(images, low - 2 * (high - low), high + 2 * (high - low))


def scale_images(images):
    """Each channel divided by 3 x its mean absolute value over the plants and pixels.

    images: (n, channels, height, width). The mean absolute value, not the mean, because the crop
    surface model (channel 0) is near zero and can be negative; with the factor 3, reflectance is
    roughly in [0, 1]."""
    images = np.asarray(images, dtype=np.float32)
    return images / (3 * np.abs(images).mean(axis=(0, 2, 3), keepdims=True))


def load_date(images_path, keep, date):
    """Clipped (clip_artifacts) and scaled (scale_images) images (n, 6, 63, 63) float32 of the kept
    plants (keep, from design) on one date (index into DATES).

    images_path: the float32 (3900, 14, 6, 63, 63) .npy written by the pipeline from
    image_archive(population)."""
    images = np.load(images_path, mmap_mode='r')
    return scale_images(clip_artifacts(images[keep, date]))
