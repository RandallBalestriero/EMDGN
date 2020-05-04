#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


__author__      = "Randall Balestriero"


def circle(N, sigma):

    DATA = np.random.randn(N, 2)
    DATA /= np.linalg.norm(DATA, 2, 1, keepdims=True)
    DATA += np.random.randn(N, 2) * sigma

    return DATA

def cosine(N, sigma):

    DATA = np.random.randn(N) * 2
    DATA = np.vstack([DATA, np.cos(DATA)]).T
    DATA += np.random.randn(N, 2) * sigma

    return DATA



