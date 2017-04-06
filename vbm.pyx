# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np


cdef extern from "vbm.h":
    void ctrain(double*, double*, int, int, double*, int, double, double, int, int, const double, const double)
    void csample(const double*, const double*, const int, const int, const int, const int, double*, double*)

def train(np.ndarray[double, ndim=2, mode='c'] W, np.ndarray[double, ndim=1, mode='c'] b, np.ndarray[double, ndim=2, mode='c'] data, int episodes, double epsilon_w, double epsilon_b, int batchsize, int seed, double momentum_constant, double decay_constant ):
    """Train a fully visible Boltzmann machine."""
    assert(W.shape[0] == W.shape[1])
    assert(W.shape[1] == b.shape[0])
    assert(data.shape[0] == episodes)
    # ctrain modifies data vector, so we need to create a copy here
    cdef np.ndarray[double, ndim=2, mode='c'] _data = data.copy()
    ctrain(<double*>W.data, <double*>b.data, W.shape[0], W.shape[1], <double*>_data.data, episodes, epsilon_w, epsilon_b, batchsize, seed, momentum_constant, decay_constant)

def sample(np.ndarray[double, ndim=2, mode='c'] W, np.ndarray[double, ndim=1, mode='c'] b, int episodes, int seed, np.ndarray[double, ndim=1, mode='c'] sinit, np.ndarray[double, ndim=2, mode='c'] samples):
    """Sample from the Boltzmann distribution defined by W and b."""
    assert(W.shape[0] == W.shape[1])
    assert(W.shape[1] == b.shape[0])
    assert(samples.shape[0] == episodes)
    assert(samples.shape[1] == b.shape[0])
    assert(sinit.shape[0] == b.shape[0])
    # csample modifies sinit vector, so we need to create a copy here
    cdef np.ndarray[double, ndim=1, mode='c'] _s = sinit.copy()
    csample(<double*>W.data, <double*>b.data, W.shape[0], W.shape[1], episodes, seed, <double*>_s.data, <double*>samples.data)
