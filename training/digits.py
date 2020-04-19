import sys
sys.path.insert(0, "../../SymJAX")
sys.path.insert(0, "../")

import numpy as np
import symjax as sj
import symjax.tensor as T
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
import matplotlib
from matrc import *
from sklearn import datasets


def plt_state():
    predictions = np.array([f(z)[0] for z in np.random.randn(100, Ds[0])])
    
    plt.figure(figsize=(30, 30))
    for i in range(50):
        plt.subplot(10, 10, 1 + i)
        plt.imshow(predictions[i].reshape((8, 8)))
    for i in range(50):
        plt.subplot(10, 10, 51 + i)
        plt.imshow(DATA[i].reshape((8, 8)))





np.random.seed(int(sys.argv[-1]) + 10)

Ds = [1, 16, 64]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])
sigma_x = 1
Sigma_x = np.eye(Ds[-1])
R, BS = 110, 150

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

batch_in_signs = T.Placeholder((R, np.sum(Ds[1:-1])), 'bool')
x = T.Placeholder((BS, Ds[-1]), 'float32')
m0 = T.Placeholder((BS, R), 'float32')
m1 = T.Placeholder((BS, R, Ds[0]), 'float32')
m2 = T.Placeholder((BS, R, Ds[0], Ds[0]), 'float32')

f, g, h, all_g, train_f, sigma_x = utils.create_fns(input, in_signs, Ds, x, m0, m1, m2, batch_in_signs, sigma_x=np.log(sigma_x), sigma=1, lr=0.001)
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


DATA = datasets.load_digits().images
#DATA /= (DATA.max(1, keepdims=True) * 100)
DATA = DATA[:BS].reshape((BS, -1)) + np.random.randn(BS, Ds[-1]) * 0.1
DATA /= DATA.max()
print(DATA)
L = []
for iter in tqdm(range(16)):

    plt_state()
    plt.savefig('samples_{}.png'.format(iter))
    plt.close()


    output, A, b, inequalities, signs = f(np.random.randn(Ds[0])/4)
    regions = utils.search_region(all_g, g, signs)
    m00, m10, m20 = [], [], []
    print('sigma', sigma_x.get({}))
    for x in DATA:
        a, b, c = utils.marginal_moments(x, regions, Sigma_x*sigma_x.get({}), sigma_z)[1:]
        m00.append(a)
        m10.append(b)
        m20.append(c)
    m00 = np.array(m00)
    m10 = np.array(m10)
    m20 = np.array(m20)

    batch_signs = np.array(list(regions.keys()))

    PP = R - m00.shape[1]
    print('PP', PP)
    if PP:
        m00 = np.concatenate([m00, np.zeros((BS, PP))], 1)
        m10 = np.concatenate([m10, np.zeros((BS, PP, Ds[0]))], 1)
        m20 = np.concatenate([m20, np.zeros((BS, PP, Ds[0], Ds[0]))], 1)
        batch_signs = np.concatenate([batch_signs,
                                      np.zeros((PP, batch_signs.shape[1]))])
    L.append(train_f(batch_signs>0, DATA, m00, m10, m20))
    print(L[-1])
    for i in range(30):
        train_f(batch_signs>0, DATA, m00, m10, m20)
    L.append(train_f(batch_signs>0, DATA, m00, m10, m20))
    print(L[-1])

    plt.figure()
    plt.plot(L, lw=3)
    plt.savefig('NLL.png')
    plt.close()


