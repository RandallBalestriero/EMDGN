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
import matplotlib.font_manager
from sklearn.neighbors import KernelDensity


def kde2D(x, y, bandwidth, X0, X1, Y0, Y1, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[X0:X1:100j,Y0:Y1:100j]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


Ds = [1, 4, 16, 2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

R, BS = 40, 50

if Ds[0] == 1:
    xx = np.linspace(-5, 5, 1500).reshape((-1, 1))
else:
    xx = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-3, 3, 100))
    xx = np.vstack([xx[0].flatten(), xx[1].flatten()]).T

ss = 0.1

for seed in [37, 146, 53, 187, 79]:
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
        
        predictions = model['sample'](20000)
        X0, X1 = predictions[:, 0].min(), predictions[:, 0].max()
        Y0, Y1 = predictions[:, 1].min(), predictions[:, 1].max()
        X0 -= 0.3
        X1 += 0.3
        Y0 -= 0.3
        Y1 += 0.3
        
        noise = np.random.randn(*predictions.shape) * ss

        # PLOT POSTERIOR
        cov_x = np.eye(2) * model['varx']()
        p2 = utils.posterior(xx, regions, outpute[None, :], As, Bs,
                                            np.eye(1), cov_x, model['input2signs'])
        m20, m21, m22 = utils.marginal_moments(outpute[None, :], regions, cov_x, np.eye(1))[1:]
 
        exp = np.einsum('Nns,nds->d', m21, As) + np.einsum('Nn,nd->d', m20, Bs)
        mu = m21.sum()

        muhat = np.average(xx[:,0], weights=p2)
        print(np.average(np.ones(len(p2)), weights=p2), m20.sum())
        print(np.average(xx[:,0], weights=p2), mu)
        print(np.average(xx[:,0] ** 2, weights=p2), m22.sum())

        if Ds[0] == 1:
            plt.plot(xx, p2, label=r'$p(\boldsymbol{z}|\boldsymbol{x})$')
            plt.axvline(mu, color='g', linestyle='--', label='mean')
        else:
            plt.imshow((np.array(p)).reshape((N, N)), extent=[-L, L, -L, L])
        for r in regions:
            vertices = utils.get_vertices(regions[r]['ineq'][:, :-1],
                                          regions[r]['ineq'][:, -1])
            plt.plot([vertices[0], vertices[0]], [-0.1, 0.1], color='k', linewidth=0.8)
            plt.plot([vertices[1], vertices[1]], [-0.1, 0.1], color='k', linewidth=0.8)
        plt.xlim([-4, 4])
        ax = plt.gca()
        ax.legend()
        plt.savefig('images/prior_{}_{}.png'.format(seed, i+ int(ss > 0.05)))
        plt.close()

        # PLOT DATA

        plt.figure(figsize=(5, 5))
        plt.scatter(predictions[::20,0] + noise[::20, 0],
                predictions[::20, 1] + noise[::20, 1], color='blue',
                    alpha=0.2)
        plt.scatter(predictions[::20,0], predictions[::20, 1], color='red',
                    alpha=0.2)
        plt.scatter(outpute[0], outpute[1], color='k',
                    label=r'$\boldsymbol{x}$')
#        plt.scatter(exp[0], exp[1], color='g', marker='x',
#                    label=r'$\mathbb{E}_{\boldsymbol{z}|\boldsymbol{x}}\left[\boldsymbol{x}\right]$')

        ax = plt.gca()
        ax.legend()
        plt.xlim([X0, X1])
        plt.ylim([Y0, Y1])
        plt.savefig('images/samples_{}_{}.png'.format(seed, i+ int(ss > 0.05)))
        plt.close()

        # PLOT DISTRIBTUION
        N = 30
        
        xxx = np.meshgrid(np.linspace(X0, X1, N), np.linspace(Y0, Y1, N))
        xxx = np.hstack([xxx[0].flatten()[:, None], xxx[1].flatten()[:, None]])
        px, m20, m21, m22 = utils.marginal_moments(xxx, regions, cov_x, np.eye(1))
        p = np.exp(px).reshape((N, N))
        fig = plt.figure()
        plt.imshow(p, extent=[X0, X1, Y0, Y1])
        plt.contour(np.linspace(X0, X1, N), np.linspace(Y0, Y1, N), 0.7**p, 4,
                    linewidths = 0.45, colors = 'w')
        cmap = matplotlib.cm.get_cmap('plasma')
        elements = [Patch(facecolor=cmap(0), edgecolor='k', label='0'),
                    Patch(facecolor=cmap(1.), edgecolor='k',
                          label=str(np.round(p.max(),2)))]
        ax = plt.gca()
        ax.legend(handles=elements)
    
        #formatit(X0, X1, Y0, Y1)
        plt.xlim([X0, X1])
        plt.ylim([Y0, Y1])
        plt.savefig('images/proba_{}_{}.png'.format(seed, i + int(ss > 0.05)))
 
        xxxx, yyyy, zzzz = kde2D(predictions[:,0] + noise[:, 0],
                            predictions[:,1] + noise[:, 1], 0.05, X0=X0, X1=X1,
                            Y0=Y0, Y1=Y1)
        print(zzzz.shape)
        plt.imshow(zzzz.T, extent=[X0, X1, Y0, Y1])
        plt.xlim([X0, X1])
        plt.ylim([Y0, Y1])
        plt.savefig('images/trueproba_{}_{}.png'.format(seed, i + int(ss > 0.05)))
 
