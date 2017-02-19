# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np


cdef extern from "vbm.h":
    void ctrain(double*, double*, int, int, double*, int, double, double, int, int)

def train(np.ndarray[double, ndim=2, mode='c'] W, np.ndarray[double, ndim=1, mode='c'] b, np.ndarray[double, ndim=2, mode='c'] data, int episodes, double epsilon_w, double epsilon_b, int batchsize, int seed):
    """Train a fully visible Boltzmann machine."""
    assert(W.shape[0] == W.shape[1])
    assert(W.shape[1] == b.shape[0])
    assert(data.shape[0] == episodes)
    # ctrain modifies data vector, so we need to create a copy here
    cdef np.ndarray[double, ndim=2, mode='c'] _data = data.copy()
    ctrain(<double*>W.data, <double*>b.data, W.shape[0], W.shape[1], <double*>_data.data, episodes, epsilon_w, epsilon_b, batchsize, seed)
