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

from matrc import *

Ds = [1, 16, 16, 2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

R, BS = 40, 50


if Ds[0] == 1:
    xx = np.linspace(-3, 3, 500).reshape((-1, 1))
else:
    xx = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    xx = np.vstack([xx[0].flatten(), xx[1].flatten()]).T



for ss in [0.1, 0.2, 0.5]:
    np.random.seed(int(sys.argv[-1]) + 10)
    for i in range(5):

        fig = plt.figure(figsize=(5,5))
        model = networks.create_fns(BS, R, Ds, 0, var_x=ss**2)

        z = np.random.randn(Ds[0])
        output, A, b, inequalities, signs = model['input2all'](z)
        outpute = output + np.random.randn(Ds[-1]) * np.sqrt(ss)
    
        regions = utils.search_region(model['signs2q'], model['signs2Ab'], signs)
        print(len(regions))
    
        As = np.array([regions[s]['Ab'][0] for s in regions])
        Bs = np.array([regions[s]['Ab'][1] for s in regions])
        
        predictions = model['sample'](200)
        noise = np.random.randn(*predictions.shape) * ss

        vx = np.eye(2) * model['varx']()
        p1 = utils.posterior(xx, regions, output, As, Bs, np.eye(1), vx, model['input2signs'])
        p2 = utils.posterior(xx, regions, outpute, As, Bs, np.eye(1), vx, model['input2signs'])
        print(p1, p2)
        print((p1 * 6 / 500).sum(), (p2 * 6 / 500).sum())

        if Ds[0] == 1:
            plt.plot(xx, p1, label=r'$p(\mathbf{z}|g(\mathbf{z}_0))$')
            plt.plot(xx, p2, label=r'$p(\mathbf{z}|g(\mathbf{z}_0)+\epsilon_0)$')
            plt.axvline(z, color='k', label=r'$\mathbf{z}_0$')
        else:
            plt.imshow((np.array(p)).reshape((N, N)), aspect='auto',
                        extent=[-L, L, -L, L], origin='lower')
        ax = plt.gca()
        ax.legend()
        plt.tight_layout()
        plt.savefig('images/prior_{}_{}.png'.format(str(ss).replace('.', ''),i))
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.scatter(predictions[:,0] + noise[:, 0],
                    predictions[:, 1] + noise[:, 1], color='blue', edgecolor='b',
                    alpha=0.2)
        plt.scatter(predictions[:,0], predictions[:, 1], color='red', edgecolor='r',
                    alpha=0.2)
        plt.scatter(output[0], output[1], color='k', marker='x', linewidth=4,
                    label=r'$g(\mathbf{z}_0)$')
        plt.scatter(outpute[0], outpute[1], color='k', linewidth=4,
                    label=r'$g(\mathbf{z}_0)+\epsilon_0$')
    
        ax = plt.gca()
        ax.legend()
        plt.tight_layout()
        plt.savefig('images/samples_{}_{}.png'.format(str(ss).replace('.', ''),i))
        plt.close()
