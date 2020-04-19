import sys
sys.path.insert(0, "../../SymJAX")
sys.path.insert(0, "../")

import numpy as np
import symjax as sj
import symjax.tensor as T
import matplotlib.pyplot as plt
import utils
import networks
from tqdm import tqdm
import matplotlib
from matrc import *



def plt_state():
    predictions = np.array([f(z)[0] for z in np.random.randn(200, Ds[0])])
    
    N = 7
    X0, X1 = predictions[:, 0].min(), predictions[:, 0].max()
    Y0, Y1 = predictions[:, 1].min(), predictions[:, 1].max()
    X0 -= 0.5
    X1 += 0.5
    Y0 -= 0.5
    Y1 += 0.5
    xxx = np.meshgrid(np.linspace(X0, X1, N), np.linspace(Y0, Y1, N))
    xxx = np.hstack([xxx[0].flatten()[:, None], xxx[1].flatten()[:, None]])
    
    p = list()
    varx = np.eye(Ds[-1]) * sigma2.get({})
    for x in tqdm(xxx):
        p.append(utils.marginal_moments(x, regions, varx,
                                        np.eye(Ds[0]))[0])
    p = np.array(p).reshape((N, N))
    
    plt.figure(figsize=(10,5))
    plt.subplot(131)
    plt.imshow(np.exp(p), extent=[X0, X1, Y0, Y1])
    plt.subplot(132)
    plt.imshow(p, extent=[X0, X1, Y0, Y1])
    plt.subplot(133)
    plt.scatter(predictions[:,0], predictions[:, 1])
    plt.scatter(DATA[:,0], DATA[:, 1], color='k')




np.random.seed(int(sys.argv[-1]) + 10)

Ds = [1, 12, 2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])
sigma_x = 1
Sigma_x = np.eye(Ds[-1])
R, BS = 110, 200

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

batch_in_signs = T.Placeholder((R, np.sum(Ds[1:-1])), 'bool')
x = T.Placeholder((BS, Ds[-1]), 'float32')
m0 = T.Placeholder((BS, R), 'float32')
m1 = T.Placeholder((BS, R, Ds[0]), 'float32')
m2 = T.Placeholder((BS, R, Ds[0], Ds[0]), 'float32')

f, g, h, all_g, train_f, sigma2 = utils.create_fns(input, in_signs, Ds, x, m0, m1, m2, batch_in_signs, sigma_x=np.log(1))

output, A, b, inequalities, signs = f(np.random.randn(Ds[0])/4)
regions = utils.search_region(all_g, g, signs) 
assert len(regions) <= R

As = np.array([regions[s]['Ab'][0] for s in regions])
Bs = np.array([regions[s]['Ab'][1] for s in regions])


N = 140
L = 4
if Ds[0] == 1:
    xx = np.linspace(-L, L, N)
    xxflag = np.linspace(-L, L, 10)
else:
    xx = np.meshgrid(np.linspace(-L, L, N), np.linspace(-L, L, N))
    xx = np.vstack([xx[0].flatten(), xx[1].flatten()]).T


DATA = np.random.randn(BS, Ds[-1])
DATA /= np.linalg.norm(DATA, 2, 1, keepdims=True)

#DATA = np.linspace(-3, 3, BS)
#DATA = np.vstack([DATA, np.cos(DATA*2)]).T
DATA += np.random.randn(BS, Ds[-1]) * 0.1

plt_state()
plt.savefig('baseline.png')
plt.close()



L = []
for iter in tqdm(range(116)):
    S, D = Ds[0], Ds[-1]
    z = np.random.randn(S)/10
    output, A, b, inequalities, signs = f(z)
    
    regions = utils.search_region(all_g, g, signs)
    print('regions', len(regions))

    varx = np.eye(D) * sigma2.get({})
    print('varx', varx)
    m0 = np.zeros((DATA.shape[0], len(regions)))
    m1 = np.zeros((DATA.shape[0], len(regions), S))
    m2 = np.zeros((DATA.shape[0], len(regions), S, S))

    for i, x in enumerate(DATA):
        m0[i], m1[i], m2[i] = utils.marginal_moments(x, regions, varx,
                                                     np.eye(S))[1:]

    P = R - len(regions)
    assert P >= 0
    m0 = np.pad(m0, [[0, 0], [0, P]])
    m1 = np.pad(m1, [[0, 0], [0, P], [0, 0]])
    m2 = np.pad(m2, [[0, 0], [0, P], [0, 0], [0, 0]])
    batch_signs = np.pad(np.array(list(regions.keys())), [[0, P], [0, 0]])
    
    m_loss = []
    for i in range(300):
        m_loss.append(train_f(batch_signs, DATA, m0, m1, m2))

    print(m_loss[0], m_loss[-1])
    

plt.figure()
plt.plot(L, lw=3)
plt.savefig('NLL.png')
plt.close()

plt_state()
plt.savefig('after.png')
plt.close()

