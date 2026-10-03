"""The paper's simulation: three heritable traits encoded into simulated sorghum spectra.

Each of the 10 instances has 3 traits with heritability 0.6, 0.4 and 0.2, each with 100 additive
QTNs of its own (effects U(-1, 1)), simulated with simplePHENOTYPES on 300 random SNPs of the 836
sequenced sorghum lines, 2 replicates per line (n = 1,672). The simplePHENOTYPES outputs are the
raw data of this simulation (the SNP sampling and the QTN effects were not seeded, so they cannot
be regenerated). Here the traits are encoded into spectra: together with 2 random latent traits
they fill the 5 latent dimensions of an autoencoder trained on the sorghum spectra, in a random
order, and are decoded.
"""

import pickle
import types

import numpy as np
import torch
import torch.nn as nn
from h2opt import AutoEncoder

from .paths import data

N_INSTANCES = 10
N_TRAITS = 3
HERITABILITY = (0.6, 0.4, 0.2)
N_QTNS = 100  # per trait
N_LATENT = 5

# Ways to get synthetic traits, by the slug used in file names, with their figure labels. 'truth'
# are the simulated traits themselves.
METHODS = {
    'truth': 'ground truth',
    'h2opt_rmsprop': 'H2-opt',
    'h2opt_lbfgs': 'H2-opt (L-BFGS)',
    'pca': 'PCA',
    'genetic_pca': 'genetic PCA',
    'lda': 'LDA',
    'pch': 'PCH',
    'max_features': 'single trait',
}
# Baselines of h2opt_analysis.evaluate.fit_baseline, by slug.
BASELINES = {'pca': 'PCA', 'genetic_pca': 'genetic PCA', 'lda': 'LDA', 'pch': 'PCH',
             'max_features': 'max features'}


def folder(instance):
    """Raw simplePHENOTYPES output of an instance."""
    return data('plant', 'simulations', 'simPhen', 'seperate100SNP', str(instance))


def load_truth(instance):
    """Plant names (line of each plant, n) and the true traits (n, 3) of an instance."""
    path = folder(instance) / 'Simulated_Data_2_Reps_Herit_0.6_0.4_0.2.txt'
    table = np.loadtxt(path, dtype=str, delimiter='\t', skiprows=1)
    return table[:, 0], table[:, 1:1 + N_TRAITS].astype(float)


def qtns(instance):
    """Chromosome and position of the QTNs, (3, 100, 2): trait t has the QTNs in rows
    100t..100t+99 of Additive_Selected_QTNs.txt (simplePHENOTYPES keeps the effect order)."""
    table = np.loadtxt(folder(instance) / 'Additive_Selected_QTNs.txt', dtype=str, delimiter='\t')
    header, rows = list(table[0]), table[1:]
    located = rows[:, [header.index('chr'), header.index('pos')]].astype(np.int64)
    return located.reshape(N_TRAITS, N_QTNS, 2)


def line_index(instance, names):
    """Row of each plant's line in the .fam of GENOTYPES.

    The genotype table given to simplePHENOTYPES has its lines in .fam order; 105 of them carry a
    synonym of the .fam ID (nameKey_corrected.tsv), so lines are matched by position, not name.
    """
    path = data('plant', 'SNP', 'simulation', 'specialSNP', f'random300SNP_{instance}.csv')
    with open(path) as f:
        columns = np.array(f.readline().strip().split(',')[5:])
    order = {name: i for i, name in enumerate(columns)}
    return np.array([order[name] for name in names])


class _PickledAutoEncoder(nn.Module):
    """Stand-in for the class of the pickled paper autoencoder, to read its weights."""


def load_paper_autoencoder(path):
    """The paper's autoencoder (a pickled model of the original code) as an h2opt AutoEncoder."""
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'autoEncoder':
                return _PickledAutoEncoder
            return super().find_class(module, name)

    pickle_module = types.SimpleNamespace(Unpickler=Unpickler, load=pickle.load,
                                          __name__='pickle')
    state = torch.load(path, weights_only=False, pickle_module=pickle_module).state_dict()
    n_features, hidden = state['linE1.weight'].shape[1], state['linE1.weight'].shape[0]
    model = AutoEncoder(n_features, state['linE2.weight'].shape[0], hidden)
    model.load_state_dict(state)
    return model


def encode_instances(truths, autoencoder, reference, seed=0):
    """Simulated spectra of each instance, as in the paper (one random stream over the instances).

    truths: list of (n, 3) true traits. reference: real spectra whose encodings set the mean and
    scale of each latent dimension. Each trait is standardized; the n x 5 latent matrix is
    N(0, 1) with the traits placed in a random permutation of the dimensions; it is rescaled to
    the reference encodings and decoded. Returns a list of (spectra (n, m), dimensions (3,)), where
    dimensions[t] is the latent dimension that holds trait t.
    """
    with torch.no_grad():
        encoded = autoencoder.encode(torch.tensor(reference).float()).numpy()
    means = np.mean(encoded, axis=0)
    stds = np.mean((encoded - means) ** 2, axis=0) ** 0.5

    rng = np.random.RandomState(seed)
    out = []
    for Y in truths:
        Y = Y - Y.mean(axis=0)
        Y = Y / np.mean(Y ** 2, axis=0) ** 0.5
        latent = rng.normal(size=len(Y) * N_LATENT).reshape((len(Y), N_LATENT))
        dimensions = rng.permutation(N_LATENT)[:Y.shape[1]]
        latent[:, dimensions] = Y
        latent = latent * stds + means
        with torch.no_grad():
            spectra = autoencoder.decode(torch.tensor(latent).float()).numpy()
        out.append((spectra, dimensions))
    return out


def test_lines(names_per_instance, seed=0):
    """Test plants of each instance, as in the paper: one random stream over the instances, a third
    of the lines (value 0 of randint(3) per line) are test lines."""
    rng = np.random.RandomState(seed)
    out = []
    for names in names_per_instance:
        _, inverse = np.unique(names, return_inverse=True)
        out.append(rng.randint(3, size=inverse.max() + 1)[inverse] == 0)
    return out


def line_means(Y, lines, n_lines):
    """Mean of each standardized column of Y over the plants of each line, (n_lines, k)."""
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    sums = np.zeros((n_lines, Y.shape[1]))
    np.add.at(sums, lines, Y)
    counts = np.bincount(lines, minlength=n_lines)
    with np.errstate(invalid='ignore'):
        return sums / counts[:, None]
