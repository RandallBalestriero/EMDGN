#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt

__author__      = "Randall Balestriero"

dataset = 'circle'

losses = []
results = []
samples = []

LEAKINESS = [0., 0.1, -1.0]
DEPTH = [1]
WIDTH = [6, 12]
SEED = [0]
for leakiness in LEAKINESS:
    for depth in DEPTH:
        for width in WIDTH:
            for seed in SEED:
                filename = "saver_{}_{}_{}_{}_{}.pkl".format(depth, width,
                                                       dataset, leakiness, seed)
                f = open(filename, 'rb')
                l, r, s = pickle.load(f)
                losses.append(l)
                results.append(r)
                samples.append(s)
                f.close()

results = np.array(results).reshape((len(LEAKINESS), len(DEPTH), len(WIDTH), len(SEED), 2, 4))
samples = np.array(samples).reshape((len(LEAKINESS), len(DEPTH), len(WIDTH), len(SEED), 2, 4, 200, 2))

for i in range(3):
    for j in range(4):
        plt.subplot(3, 4, 1 + 4 * i + j)
        plt.scatter(samples[i, 0, 0, 0, 1, j, :, 0], samples[i, 0, 0, 0, 1, j, :, 1])

plt.show()
print(results)
    
