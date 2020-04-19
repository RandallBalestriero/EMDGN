import sys
sys.path.insert(0, "../../SymJAX")
sys.path.insert(0, "../")
import numpy as np
import symjax as sj
import symjax.tensor as T
import cdd
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from scipy.spatial import ConvexHull
from multiprocessing import Pool
from matplotlib.patches import Patch


from matrc import *
import matplotlib

Ds = [2, 6, 6, 2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

R, BS = 40, 50

batch_in_signs = T.Placeholder((R, np.sum(Ds[1:-1])), 'bool')
x = T.Placeholder((BS, Ds[-1]), 'float32')
m0 = T.Placeholder((BS, R), 'float32')
m1 = T.Placeholder((BS, R, Ds[0]), 'float32')
m2 = T.Placeholder((BS, R, Ds[0], Ds[0]), 'float32')

if Ds[0] == 1:
    xx = np.linspace(-3, 3, 500)
else:
    L = 3
    N = 100
    xx = np.meshgrid(np.linspace(-L, L, N), np.linspace(-L, L, N))
    xx = np.vstack([xx[0].flatten(), xx[1].flatten()]).T



for ss in [0.05, 0.2, 0.5]:
    np.random.seed(int(sys.argv[-1]) + 10)
    sigma_x = np.eye(Ds[-1]) * ss
    for i in range(20):
        fig = plt.figure(figsize=(5,5))
        f, g, h, all_g, train_f = utils.create_fns(input, in_signs, Ds, x, m0,
                                                m1, m2, batch_in_signs, sigma=2)

        z = np.random.randn(Ds[0])
        output, A, b, inequalities, signs = f(z)
        outpute = output + np.random.randn(Ds[-1]) * np.sqrt(0.05)
    
        regions = utils.search_region(all_g, g, signs)
        print(len(regions))
    
        As = np.array([regions[s]['Ab'][0] for s in regions])
        Bs = np.array([regions[s]['Ab'][1] for s in regions])
        
        predictions = np.array([f(np.random.randn(Ds[0]))[0]
                                for z in range(200)])
        noise = np.random.randn(*predictions.shape)*np.sqrt(sigma_x[0,0])

        p1 = utils.posterior(xx, regions, output, As, Bs, mu_z, sigma_z,
                             sigma_x)
        p2 = utils.posterior(xx, regions, outpute, As, Bs, mu_z, sigma_z,
                             sigma_x)

        p1 = np.array(p1).reshape((N, N))
        p2 = np.array(p2).reshape((N, N))

        plt.imshow(p1, extent=[-L, L, -L, L])
        plt.contour(np.linspace(-L, L, N), np.linspace(-L, L, N), p1, 6, 
                    linewidths = 0.35, colors = 'w')
 
        ll = plt.scatter(z[0], z[1], color='k', label=r'$\mathbf{z}_0$')
        ax = plt.gca()
        cmap = matplotlib.cm.get_cmap('plasma')
        elements = [Patch(facecolor=cmap(0), edgecolor='k', label=str(np.round(p1.min(),2))),
                    Patch(facecolor=cmap(1.), edgecolor='k',
                          label=str(np.round(p1.max(),2))), ll]
        ax = plt.gca()
        ax.legend(handles=elements)
        plt.savefig('images/two_prior1_{}_{}.png'.format(str(ss).replace('.', ''),i))
        plt.close()

        plt.figure()
        plt.imshow(p2, extent=[-L, L, -L, L])
        plt.contour(np.linspace(-L, L, N), np.linspace(-L, L, N), p1, 6,
                linewidths = 0.35, colors = 'w')
        ll = plt.scatter(z[0], z[1], color='k', label=r'$\mathbf{z}_0$')
        ax = plt.gca()
        cmap = matplotlib.cm.get_cmap('plasma')
        elements = [Patch(facecolor=cmap(0), edgecolor='k', label=str(np.round(p2.min(),2))),
                    Patch(facecolor=cmap(1.), edgecolor='k',
                          label=str(np.round(p2.max(),2))), ll]
        ax = plt.gca()
        ax.legend(handles=elements)
        plt.savefig('images/two_prior2_{}_{}.png'.format(str(ss).replace('.', ''),i))
        plt.close()


        plt.figure()
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
        plt.savefig('images/two_samples_{}_{}.png'.format(str(ss).replace('.', ''),i))
        plt.close()
