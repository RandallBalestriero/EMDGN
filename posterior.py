import sys
sys.path.insert(0, "../SymJAX")
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

np.random.seed(int(sys.argv[-1]) + 10)

Ds = [1, 32, 16, 2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])
sigma_x = np.eye(Ds[-1]) * 0.05

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

R, BS = 40, 50

batch_in_signs = T.Placeholder((R, np.sum(Ds[1:-1])), 'bool')
x = T.Placeholder((BS, Ds[-1]), 'float32')
m0 = T.Placeholder((BS, R), 'float32')
m1 = T.Placeholder((BS, R, Ds[0]), 'float32')
m2 = T.Placeholder((BS, R, Ds[0], Ds[0]), 'float32')


plt.figure(figsize=(16,4))
for i in range(C):
    f, g, h, all_g, train_f = utils.create_fns(input, in_signs, Ds, x, m0, m1, m2,
            batch_in_signs, sigma=2)

    regions = utils.search_region(all_g, g, signs)

    As = np.array([regions[s]['Ab'][0] for s in regions])
    Bs = np.array([regions[s]['Ab'][1] for s in regions])
    
    output, A, b, inequalities, signs = f(np.random.randn(Ds[0]))
    
    predictions = np.array([f(np.random.randn(Ds[0]))[0] for z in range(200)])

    plt.subplot(121)
    noise = np.random.randn(*predictions.shape)*np.sqrt(sigma_x[0, 0])
    plt.scatter(predictions[:,0] + noise[:, 0],
                predictions[:, 1] + noise[:, 1], color='blue',
                label=r'Noisy output $g(\mathbf{z})+\epsilon$')
    plt.scatter(predictions[:,0], predictions[:, 1], color='red',
                label=r'Noiseless output $g(\mathbf{z})$')
    plt.legend()

    N = 35
    X0, X1 = predictions[:, 0].min(), predictions[:, 0].max()
    Y0, Y1 = predictions[:, 1].min(), predictions[:, 1].max()
    X0 -= 0.5
    X1 += 0.5
    Y0 -= 0.5
    Y1 += 0.5
    
    plt.xlim([X0, X1])
    plt.ylim([Y0, Y1])

    xxx = np.meshgrid(np.linspace(X0, X1, N), np.linspace(Y0, Y1, N))
    xxx = np.hstack([xxx[0].flatten()[:, None], xxx[1].flatten()[:, None]])
    
    p = list()
    for x in tqdm(xxx):
        p.append(utils.marginal_moments(x, regions, sigma_x, mu_z, sigma_z)[0])
    p = np.array(p).reshape((N, N))

    plt.subplot(122)
    plt.imshow(np.exp(p), aspect='auto', extent=[X0, X1, Y0, Y1],
                origin='lower')
    plt.colorbar()


    plt.savefig('test1_{}.png'.format(sys.argv[-1]))
