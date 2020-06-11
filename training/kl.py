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

R = 3


LVAE = []
LEM = []

for lr in [0.005, 0.001, 0.0001]:
    for run in range(R):
        data = np.load('nnnnsaving_likelihood_{}_150_VAE_{}_{}_{}.npz'.format(dataset, lr, network, run))
        LVAE.append(data['LL'][:, -1])
        LEM.append(data['L'][1:])
        print(LEM[-1].shape)
        print(LVAE[-1].shape)
 
LVAE = np.array(LVAE).reshape((3, R, -1))
LEM = np.array(LEM).reshape((3, R, -1))


fig, ax = plt.subplots(1, 3, sharey='row', figsize=(6,3))
colors = ['b', 'r', 'g']
for e, c in enumerate(colors):
    for l, m in zip(LVAE[e], LEM[e]):
        print('VAE',-l[-1],'EM', -m[-1])
        ax[e].plot(-l+m, c=c,lw=1,alpha=1)

ax[0].set_xticks([])
ax[1].set_xticks([])
ax[2].set_xticks([])

ax[0].set_ylabel(r'$KL(q(\mathbf{x})||p(\mathbf{z}|\mathbf{x}))$', fontsize=20)
plt.tight_layout()
plt.savefig('KL_{}_{}.png'.format(dataset, network))
asdf

   
