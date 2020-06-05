#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

file = sys.argv[-1]


__author__      = "Randall Balestriero"

dataset = 'circle'
network = 'large'

R = 5


if '.py' not in file:
    data = np.load(file)
    if 'mnist' in file:
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(data['samples'][i].reshape((28, 28)))
    else:
        plt.plot(data['samples'][:,0], data['samples'][:,1], 'x')
        plt.plot(data['data'][:,0], data['data'][:,1], 'x')
    if data['L'].ndim == 2:
        print(data['L'].mean(1))
    else:
        print(data['L'])
    plt.show()

    if data['L'].ndim == 2:
        plt.plot(data['L'].mean(1))
    else:
        plt.plot(data['L'])
    plt.show()


LEM = []
LVAE = []

for run in range(R):
    LEM.append(np.load('nnsaving_likelihood_{}_100_EM_{}_{}.npz'.format(dataset, network, run))['L'])

#for lr in [0.005, 0.001, 0.0001]:
#    for run in range(R):
#        LVAE.append(np.load('nnsaving_likelihood_{}_100_VAE_{}_{}_{}.npz'.format(dataset, lr, network, run))['L'])
#        if LVAE[-1].ndim ==2:
#            LVAE[-1] = LVAE[-1].mean(1)
 
LEM = np.array(LEM).reshape((R, -1))
#LVAE = np.array(LVAE).reshape((3, R, -1))

for l in LEM:
    plt.plot(l,'k')

#colors = ['b', 'r', 'g']
#for e, c in enumerate(colors):
#    for l in LVAE[e]:
#        plt.plot(l, c=c)

plt.show()
asdf


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
    
