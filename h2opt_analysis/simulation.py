"""The paper's simulation: three heritable traits encoded into simulated sorghum spectra.

Each of the 10 instances has 3 traits with heritability 0.6, 0.4 and 0.2, each with 100 additive
QTNs of its own (effects U(-1, 1)), simulated with simplePHENOTYPES on 300 random SNPs of the 836
sequenced sorghum lines, 2 replicates per line (n = 1,672). The simplePHENOTYPES outputs are the
raw data of this simulation (the SNP sampling and the QTN effects were not seeded, so they cannot
be regenerated). Here the traits are encoded into spectra: together with 2 random latent traits
they fill the 5 latent dimensions of an autoencoder trained on the sorghum spectra, in a random
order, and are decoded with the decoder linearized at the latent means (h2opt.encode_latent). The
paper decoded with the full tanh decoder; its products and squares of the traits are heritable and
appeared as extra synthetic traits. Each plant also gets the autoencoder residual of a random real
sorghum spectrum (h2opt.reconstruction_residuals), the real variation that the 5 latent dimensions
miss (0.7% of the real spectra's variance), which is not heritable in the simulation.
"""

import functools
import pickle
import types

import numpy as np
import torch
import torch.nn as nn
from h2opt import (
    AutoEncoder,
    ConvModel,
    ImageConvModel,
    encode_latent,
    latent_scale,
    reconstruction_residuals,
)
from h2opt.folds import per_group

from .paths import data

N_INSTANCES = 10
N_TRAITS = 3
HERITABILITY = (0.6, 0.4, 0.2)
N_QTNS = 100  # per trait
N_LATENT = 5
PAPER_AUTOENCODER = data('plant', 'models', 'autoencode_2.pt')  # pickled model of the paper code


def make_cnn(X):
    """The CNN of the hybrid H2-opt for the measurements X (plants first), a function with no
    arguments: h2opt.ConvModel over the features of a spectrum (X: plants x features), or
    h2opt.ImageConvModel over images (X: plants x channels x height x width)."""
    if X.ndim == 2:
        return functools.partial(ConvModel, X.shape[1])
    return functools.partial(ImageConvModel, X.shape[1])


def correlations(truth, Y):
    """Pearson correlation of each column of truth with each column of Y, (k_truth, k_Y)."""
    a = (truth - truth.mean(0)) / truth.std(0)
    b = (Y - Y.mean(0)) / Y.std(0)
    return a.T @ b / len(a)


def folder(instance):
    """Raw simplePHENOTYPES output of an instance."""
    return data('plant', 'simulations', 'simPhen', 'seperate100SNP', str(instance))


def truth_file(instance):
    """simplePHENOTYPES table of the plants of an instance (the folders also hold other runs)."""
    return folder(instance) / 'Simulated_Data_2_Reps_Herit_0.6_0.4_0.2.txt'


def load_truth(instance):
    """Plant names (line of each plant, n) and the true traits (n, 3) of an instance."""
    table = np.loadtxt(truth_file(instance), dtype=str, delimiter='\t', skiprows=1)
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
    scale of each latent dimension (h2opt.latent_scale). Each trait is standardized; the n x 5
    latent matrix is N(0, 1) with the traits placed in a random permutation of the dimensions, and
    is decoded by h2opt.encode_latent (linearized decoder), plus the residual of a random
    spectrum of reference (h2opt.reconstruction_residuals), drawn by a second random stream (seed
    + 1) so that the latents are those of the paper. Returns a list of (spectra (n, m),
    dimensions (3,)), where dimensions[t] is the latent dimension that holds trait t.
    """
    scale = latent_scale(autoencoder, reference)
    residuals = reconstruction_residuals(autoencoder, reference)
    rng = np.random.RandomState(seed)
    residual_rng = np.random.RandomState(seed + 1)
    out = []
    for Y in truths:
        Y = Y - Y.mean(axis=0)
        Y = Y / np.mean(Y ** 2, axis=0) ** 0.5
        latent = rng.normal(size=len(Y) * N_LATENT).reshape((len(Y), N_LATENT))
        dimensions = rng.permutation(N_LATENT)[:Y.shape[1]]
        latent[:, dimensions] = Y
        out.append((encode_latent(latent, autoencoder, scale, residuals, residual_rng), dimensions))
    return out


def line_folds(names_per_instance, seed=0):
    """Fold (0, 1 or 2) of each plant of each instance, as in the paper: one random stream over the
    instances, randint(3) per line. Fold 0 are the test lines (test_lines)."""
    rng = np.random.RandomState(seed)
    return [per_group(names, lambda c: rng.randint(3, size=c)) for names in names_per_instance]


def test_lines(names_per_instance, seed=0):
    """Test plants of each instance, as in the paper: a third of the lines (fold 0 of
    line_folds)."""
    return [folds == 0 for folds in line_folds(names_per_instance, seed)]


def instance_folds(instance):
    """line_folds of one instance (the stream runs over all instances)."""
    return line_folds([load_truth(i)[0] for i in range(N_INSTANCES)])[instance]
