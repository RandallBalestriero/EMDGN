import sys
sys.path.insert(0, "../../SymJAX")
sys.path.insert(0, "../")
import numpy as np
import symjax as sj
import symjax.tensor as T
import cdd
import matplotlib.pyplot as plt
import utils
import networks
from tqdm import tqdm
from scipy.spatial import ConvexHull
from multiprocessing import Pool
import matplotlib
from matplotlib.patches import Patch
from matrc import *

Ds = [1, 4, 16, 2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

R, BS = 40, 50

if Ds[0] == 1:
    xx = np.linspace(-5, 5, 500).reshape((-1, 1))
else:
    xx = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-3, 3, 100))
    xx = np.vstack([xx[0].flatten(), xx[1].flatten()]).T

ss = 0.05

for seed in range(200):
    np.random.seed(seed)
    for i in range(1):

        fig = plt.figure(figsize=(5,5))
        model = networks.create_fns(BS, R, Ds, seed, var_x=np.ones(Ds[-1]) * ss**2,
                leakiness=0.01)

        z = np.random.randn(Ds[0])
        output, A, b, inequalities, signs = model['input2all'](z)
        outpute = output + np.random.randn(Ds[-1]) * ss
    
        regions = utils.search_region(model['signs2ineq'], model['signs2Ab'], signs)
        print(len(regions))
    
        As = np.array([regions[s]['Ab'][0] for s in regions])
        Bs = np.array([regions[s]['Ab'][1] for s in regions])
        
        predictions = model['sample'](200)

        noise = np.random.randn(*predictions.shape) * ss

        # PLOT POSTERIOR
        vx = np.eye(2) * model['varx']()
        p2, m20, m21, m22 = utils.posterior(xx, regions, outpute, As, Bs,
                                            np.eye(1), vx, model['input2signs'])
        exp = np.einsum('ns,nds->d', m21, As) + np.einsum('n,nd->d', m20, Bs)
        mu = m21.sum()

        muhat = np.average(xx[:,0], weights=p2)
        print(muhat, mu)
        print(np.average((xx[:,0])**2, weights=p2), m22.sum())

        if Ds[0] == 1:
            plt.plot(xx, p2, label=r'$p(\mathbf{z}|\mathbf{x})$')
            plt.axvline(mu, color='b', label='mean')
        else:
            plt.imshow((np.array(p)).reshape((N, N)), aspect='auto',
                        extent=[-L, L, -L, L], origin='lower')
        ax = plt.gca()
        ax.legend()
        plt.tight_layout()
        plt.savefig('images/prior_{}_{}.png'.format(seed, i))
        plt.close()

        # PLOT DATA

        plt.figure(figsize=(5, 5))
        plt.scatter(predictions[:,0] + noise[:, 0],
                    predictions[:, 1] + noise[:, 1], color='blue', edgecolor='b',
                    alpha=0.2)
        plt.scatter(predictions[:,0], predictions[:, 1], color='red', edgecolor='r',
                    alpha=0.2)
        plt.scatter(outpute[0], outpute[1], color='k',
                    label=r'$\mathbf{x}$', linewidth=4)
        plt.scatter(exp[0], exp[1], color='g', marker='x',  linewidth=4,
                    label=r'$\mathbb{E}_{\mathbf{z}|\mathbf{x}}\left[\mathbf{x}\right]$')



    
        ax = plt.gca()
        ax.legend()
        plt.tight_layout()
        plt.savefig('images/samples_{}_{}.png'.format(seed, i))
        plt.close()

        # PLOT DISTRIBTUION
        N = 30
        X0, X1 = predictions[:, 0].min(), predictions[:, 0].max()
        Y0, Y1 = predictions[:, 1].min(), predictions[:, 1].max()
        X0 -= 0.3
        X1 += 0.3
        Y0 -= 0.3
        Y1 += 0.3
        
        xxx = np.meshgrid(np.linspace(X0, X1, N), np.linspace(Y0, Y1, N))
        xxx = np.hstack([xxx[0].flatten()[:, None], xxx[1].flatten()[:, None]])
        
        p = list()
        cov_x = np.eye(2) * model['varx']()
        for xxxx in tqdm(xxx):
            p.append(utils.marginal_moments(xxxx, regions, cov_x, np.eye(1))[0])
        p = np.array(p).reshape((N, N))
    
        fig = plt.figure()
        plt.imshow(np.exp(p), extent=[X0, X1, Y0, Y1])
        plt.contour(np.linspace(X0, X1, N), np.linspace(Y0, Y1, N),np.exp(p), 8,
                    linewidths = 0.45, colors = 'w')
        cmap = matplotlib.cm.get_cmap('plasma')
        elements = [Patch(facecolor=cmap(0), edgecolor='k', label='0'),
                    Patch(facecolor=cmap(1.), edgecolor='k',
                          label=str(np.round(np.exp(p).max(),2)))]
        ax = plt.gca()
        ax.legend(handles=elements)
    
        #formatit(X0, X1, Y0, Y1)
        plt.xlim([X0, X1])
        plt.ylim([Y0, Y1])
        plt.savefig('images/proba_{}_{}.png'.format(seed, i))
 

