#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

file = sys.argv[-1]


__author__      = "Randall Balestriero"

dataset = 'wave'
network = 'small'

R = 5


LVAE = []
LEM = []

for lr in [0.005, 0.001, 0.0001]:
    for run in range(R):
        data = np.load('nsaving_likelihood_{}_100_VAE_{}_{}_{}.npz'.format(dataset, lr, network, run))
        LVAE.append(data['L'])
        print(LVAE[-1].shape)
        if LVAE[-1].ndim ==2:
            LVAE[-1] = LVAE[-1].mean(1)
        LEM.append(data['LL'])
        print(LEM[-1].shape)
 
LVAE = np.array(LVAE).reshape((3, R, -1))
LEM = np.array(LEM).reshape((3, R, -1))


fig, ax = plt.subplots(1, 3, sharey='row')
colors = ['b', 'r', 'g']
for e, c in enumerate(colors):
    for l, m in zip(LVAE[e], LEM[e]):
        print(m)
        ax[e].plot(np.arange(len(m)), m, c='k')
        ax[e].plot(np.linspace(0, len(m)-1, len(l)), l, c='--k')

plt.show()
asdf

   
