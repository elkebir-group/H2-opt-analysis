"""Pearson correlations of the Miscanthus synthetic traits with the measured traits.

The measured traits (MEASURED_TRAITS) are, for Central Japan, 16 measured traits from the
2020 columns of its plant data table; for South Japan, the 8 of these that its plant data table
also has (no biomass, culm or height columns). Dates are converted to days. The synthetic traits
are trait 1 of one method (one file per date: folds x kept plants x traits), on all kept plants
(training and test plants). For each measured trait, only the kept plants with
a value are used. Each correlation is computed per plant and between half-sib family means (over
these plants), in each fold. Writes an .npz (traits; plants: the plants with a value per trait;
dates; plant and family: traits x dates x folds) and prints the median |r| over the folds of the
family means for the first five dates.

    python pipelines/scripts/miscanthus/measured_correlations.py --population central \\
        --traits date0.npz ... date13.npz --out measured_correlations.npz
"""

import argparse
import datetime

import numpy as np
from date_correlations import family_means, load_traits, pearson

from h2opt_analysis import miscanthus

TABLES = {population: miscanthus.INPUTS / 'data_process' / f'Msi_{p}_Japan_2019-URB-Plant_Data.tsv'
          for population, p in (('central', 'C'), ('south', 'S'))}
# Per population: name in the figures -> column of the plant data table (Central: the first column
# of that name is 2020).
MEASURED_TRAITS = {
    'central': {
        'Biomass': 'Biomass Yield dry weight (g)',
        'Average internode length': 'Average Internode Length (AIL; cm)',
        'Culm density': 'Culm Density (mass/volume)) (CmDW/V)',
        'Culm dry weight': 'Culm Dry Weight (CmDW; g)',
        'Culm node number': 'Culm Node Number (CmN#)',
        'Basal diameter': 'Culm Outer Diameter at the Basal Internode (CmD_I1; mm)',
        'Last internode diameter': 'Culm Outer Diameter at Last internode (CmD_LI; mm)',
        'Height': 'Culm Length (CmL; __ Nov 2020; cm)',
        'Basal circumference': 'Basal Circumference (BCirc; cm)   2020',
        'Dormancy date': 'Autumn Dormancy Date 2020',
        'First heading date': 'First Heading Date 2020 (HD)',
        '50% heading date': '50% Heading Date 2020 (HD 50%)',
        'Hardiness score': 'Hardiness score (%) ( July 2020)',
        'Spring regrowth date': 'Spring Re-growth Date 2020',
        'Survival Oct. 2019': 'Survival (Oct __07-_ 2019)',
        'Survival spring 2020': 'Survival (spring 2020)',
    },
    'south': {
        'Basal circumference': 'Basal Circumference (BCirc; cm)   2020',
        'Dormancy date': 'Autumn Dormancy Date 2020',
        'First heading date': 'First Heading Date 2020 (HD)',
        '50% heading date': '50% Heading Date 2020 (HD 50%)',
        'Hardiness score': 'Hardiness score (%) ( July 2020)',
        'Spring regrowth date': 'Spring Re-growth Date 2020',
        'Survival Oct. 2019': 'Survival (Oct 04 - Oct 07 2019)',
        'Survival spring 2020': 'Survival (spring 2020)',
    },
}
MISSING = ('', '-', 'Missing', 'Missed', 'no stem', 'no stems', 'NA')


def parse_value(text):
    """A table entry as a float (a date as days since 2019-01-01); NaN if missing."""
    text = text.strip()
    if text in MISSING:
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    for pattern in ('%d-%b-%Y', '%d-%b-%y'):
        try:
            return (datetime.datetime.strptime(text, pattern) - datetime.datetime(2019, 1, 1)).days
        except ValueError:
            pass
    raise ValueError(f'cannot read the table entry {text!r}')


def measured_traits(population, keep):
    """(traits x kept plants) values of MEASURED_TRAITS[population], NaN where missing. Rows
    without a plot ID (a trailing row of the South table) are not plants."""
    path = TABLES[population]
    header = path.read_text().split('\n', 1)[0].rstrip('\r').split('\t')
    table = np.loadtxt(path, delimiter='\t', dtype=str, skiprows=1)
    table = table[table[:, 0] != '']
    plants = np.loadtxt(miscanthus.plant_table(population), delimiter='\t', dtype=str, skiprows=1)
    assert (table[:, 2] == plants[:, 2]).all() and (table[:, 3] == plants[:, 3]).all(), \
        'the plant data table is not in the order of the plant table'
    columns = [header.index(column) for column in MEASURED_TRAITS[population].values()]
    return np.array([[parse_value(v) for v in table[keep, c]] for c in columns])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--traits', nargs='+', required=True, help='one file per date')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    traits = load_traits(args.traits)  # dates x folds x plants
    n_dates, n_folds = traits.shape[:2]
    values = measured_traits(args.population, design.keep)
    names = list(MEASURED_TRAITS[args.population])
    plant, family = np.zeros((2, len(names), n_dates, n_folds))
    for t in range(len(names)):
        has = np.isfinite(values[t])
        trait_means = family_means(values[t, has], design.families[has])
        for d in range(n_dates):
            for f in range(n_folds):
                plant[t, d, f] = pearson(traits[d, f, has], values[t, has])
                family[t, d, f] = pearson(family_means(traits[d, f, has], design.families[has]),
                                          trait_means)
    counts = np.isfinite(values).sum(axis=1)
    np.savez(args.out, traits=names, plants=counts, dates=miscanthus.DATES, plant=plant,
             family=family)
    print('median |r| over folds of family means, dates ' + ' '.join(miscanthus.DATES[:5]))
    for t, name in enumerate(names):
        row = np.median(np.abs(family[t, :5]), axis=1)
        print(f'{name:28s}{counts[t]:6d}' + ''.join(f'{v:7.2f}' for v in row))


if __name__ == '__main__':
    main()
