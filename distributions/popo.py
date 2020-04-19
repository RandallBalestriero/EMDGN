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
import matplotlib
from matplotlib.patches import Patch
import networks
from matrc import *


Ds = [1, 16, 16, 2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

R, BS = 40, 50

def formatit(X0, X1, Y0, Y1):
    w = list(range(int(np.ceil(X0)), int(X1) + 1))
    plt.xticks(w, [str(t) for t in w])

    w = list(range(int(np.ceil(Y0)), int(Y1) + 1))
    plt.yticks(w, [str(t) for t in w])


for ss in [0.5, 0.2, 0.5]:
    np.random.seed(int(sys.argv[-1]) + 10)
    sigma_x = np.eye(Ds[-1]) * ss
    for i in range(5):
        fig = plt.figure(figsize=(5,5))
        model =  networks.create_fns(BS, R, Ds, 0, var_x = ss**2)
    
        output, A, b, inequalities, signs = model['input2all'](np.random.randn(Ds[0]))
    
        regions = utils.search_region(model['signs2q'], model['signs2Ab'], signs)
    
        As = np.array([regions[s]['Ab'][0] for s in regions])
        Bs = np.array([regions[s]['Ab'][1] for s in regions])
        
        predictions = np.array([model['input2all'](np.random.randn(Ds[0]))[0] for z in range(200)])
    
        noise = np.random.randn(*predictions.shape) * ss
        plt.scatter(predictions[:,0] + noise[:, 0],
                    predictions[:, 1] + noise[:, 1], color='blue',
                    label=r'$g(\mathbf{z})+\epsilon$', alpha=0.5,
                    edgecolors='k')
        plt.scatter(predictions[:,0], predictions[:, 1], color='red',
                    label=r'$g(\mathbf{z})$', alpha=0.5, 
                    edgecolors='k')
        ax = plt.gca()
        ax.legend()
    
        N = 25
        X0, X1 = predictions[:, 0].min(), predictions[:, 0].max()
        Y0, Y1 = predictions[:, 1].min(), predictions[:, 1].max()
        X0 -= 0.5
        X1 += 0.5
        Y0 -= 0.5
        Y1 += 0.5
        
        #formatit(X0, X1, Y0, Y1)
        plt.xlim([X0, X1])
        plt.ylim([Y0, Y1])
        plt.savefig('images/generate_{}_{}.png'.format(str(sigma_x[0, 0]).replace('.', ''), i))
        plt.close()
    
        xxx = np.meshgrid(np.linspace(X0, X1, N), np.linspace(Y0, Y1, N))
        xxx = np.hstack([xxx[0].flatten()[:, None], xxx[1].flatten()[:, None]])
        
        p = list()
        cov_x = np.eye(2) * model['varx']()
        for xx in tqdm(xxx):
            p.append(utils.marginal_moments(xx, regions, cov_x, np.eye(1))[0])
        p = np.array(p).reshape((N, N))
    
        fig = plt.figure()
        plt.imshow(np.exp(p), extent=[X0, X1, Y0, Y1])
        plt.contour(np.linspace(X0, X1, N), np.linspace(Y0, Y1, N),np.exp(p), 12, 
                    linewidths = 0.35, colors = 'w')
        cmap = matplotlib.cm.get_cmap('plasma')
        elements = [Patch(facecolor=cmap(0), edgecolor='k', label='0'),
                    Patch(facecolor=cmap(1.), edgecolor='k',
                          label=str(np.round(np.exp(p).max(),2)))]
        ax = plt.gca()
        ax.legend(handles=elements)
    
        #formatit(X0, X1, Y0, Y1)
        plt.xlim([X0, X1])
        plt.ylim([Y0, Y1])
        plt.savefig('images/proba_{}_{}.png'.format(str(sigma_x[0, 0]).replace('.',''), i))
        plt.close()
