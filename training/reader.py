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
network = 'large'

R = 15


if '.py' not in file:
    data = np.load(file)
    if 'mnist' in file:
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(data['samples'][i].reshape((28, 28)))
    else:
        plt.plot(data['samples'][:,0], data['samples'][:,1], 'x')
        plt.plot(data['data'][:,0], data['data'][:,1], 'kx')
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
SAMPLESEM = []
SAMPLESVAE = []



for run in range(R):
    EM = np.load('nnnsaving_likelihood_{}_75_EM_{}_{}.npz'.format(dataset, network, run))['L']
    print(np.diff(EM).max())
    if np.diff(EM).max() > 0.01:
        continue
    LEM.append(np.load('nnnsaving_likelihood_{}_75_EM_{}_{}.npz'.format(dataset, network, run))['L'])
    SAMPLESEM.append(np.load('nnnsaving_likelihood_{}_75_EM_{}_{}.npz'.format(dataset, network, run))['samples'])

for lr in [0.005, 0.001, 0.0001]:
    for run in range(len(LEM)):
        DATA = np.load('nnnnsaving_likelihood_{}_75_VAE_{}_{}_{}.npz'.format(dataset, lr, network, run))['data']
        plt.figure(figsize=(4,4))
        plt.plot(DATA[:, 0], DATA[:, 1], 'x')
        plt.savefig('data.png')
        LVAE.append(np.load('nnnnsaving_likelihood_{}_75_VAE_{}_{}_{}.npz'.format(dataset, lr, network, run))['L'])
        if LVAE[-1].ndim ==2:
            LVAE[-1] = LVAE[-1].mean(1)
        SAMPLESVAE.append(np.load('nnnnsaving_likelihood_{}_75_VAE_{}_{}_{}.npz'.format(dataset, lr, network, run))['samples'])
 
LEM = np.array(LEM).reshape((len(LEM), -1))
LVAE = np.array(LVAE).reshape((3, len(LEM), -1))


fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

for l in LEM:
    ax1.semilogy(l,'k',alpha=0.5, lw=3)

colors = ['b', 'r', 'g']
for e, c in enumerate(colors):
    for l in LVAE[e]:
        ax1.semilogy(l, c=c, alpha=0.5, lw=3)


#ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(np.array([0, len(LEM[0]) // 2, len(LEM[0]) - 1]))
ax2.set_xticklabels(np.array([0, len(LVAE[0,0]) * 500 // 2, len(LVAE[0,0])*500 - 1]))
ax2.set_xlabel(r"Number of VAE updates", fontsize=20)
ax2.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax1.set_ylabel(r'$- \log(p(\mathbf{x}))$', fontsize=20)
#ax1.set_yticks([])
ax1.set_xlabel(r'EM-steps', fontsize=20)
ax1.tick_params(axis = 'both', which = 'major', labelsize = 20)
#ax1.set_ylim([2.2,3.5])
plt.tight_layout()
plt.savefig('savedsemi_{}_{}.png'.format(dataset, network))
plt.close()

R = len(LEM)

if dataset == 'mnist':
    f, ax = plt.subplots(3, 9, figsize=(15,5))
    for i in range(3):
        for j in range(9):
            ax[i, j].imshow(SAMPLESEM[i][j].reshape((28, 28)))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.savefig('samples_EM_{}_{}.png'.format(dataset, network))
    plt.close()  
    
    
    f, ax = plt.subplots(3, 9, figsize=(15,5))
    for i in range(3):
        for j in range(9):
            ax[i, j].imshow(SAMPLESVAE[i][j].reshape((28, 28)))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.savefig('samples_VAE1_{}_{}.png'.format(dataset, network))
    plt.close()  
    
    
    f, ax = plt.subplots(3, 9, figsize=(15,5))
    for i in range(3):
        for j in range(9):
            ax[i, j].imshow(SAMPLESVAE[R+i][j].reshape((28, 28)))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.savefig('samples_VAE2_{}_{}.png'.format(dataset, network))
    plt.close()
    
    f, ax = plt.subplots(3, 9, figsize=(15,5))
    for i in range(3):
        for j in range(9):
            ax[i, j].imshow(SAMPLESVAE[2*R+i][j].reshape((28, 28)))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.savefig('samples_VAE3_{}_{}.png'.format(dataset, network))
    plt.close()
else:
    f, ax = plt.subplots(4, 3, figsize=(15,12))
    for i in range(3):
        ax[0, i].plot(SAMPLESEM[i][:, 0], SAMPLESEM[i][:, 1], 'x')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])

    for i in range(3):
        ax[1, i].plot(SAMPLESVAE[i][:, 0], SAMPLESVAE[i][:, 1], 'x')
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
    
    for i in range(3):
        ax[2, i].plot(SAMPLESVAE[R+i][:, 0], SAMPLESVAE[R+i][:, 1], 'x')
        ax[2, i].set_xticks([])
        ax[2, i].set_yticks([])
 
    for i in range(3):
        ax[3, i].plot(SAMPLESVAE[2*R+i][:, 0], SAMPLESVAE[2*R+i][:, 1], 'x')
        ax[3, i].set_xticks([])
        ax[3, i].set_yticks([])

    ax[0,0].set_ylabel('EM training', fontsize=16)
    ax[1,0].set_ylabel('VAE (large lr)', fontsize=16)
    ax[2,0].set_ylabel('VAE (medium lr)', fontsize=16)
    ax[3,0].set_ylabel('VAE (small lr)', fontsize=16)

    plt.tight_layout()
    
    plt.savefig('samples_all_{}_{}.png'.format(dataset, network))
 

asf

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
    
