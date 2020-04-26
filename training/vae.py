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



np.random.seed(int(sys.argv[-2]) + 10)

Ds = [1, 4, 2]
R, BS = 100, 250

if sys.argv[-1] == 'VAE':
    model = networks.create_vae(BS, Ds, 1, lr=0.005, leakiness=0.1)
else:
    model = networks.create_fns(BS, R, Ds, 1, var_x = np.ones(Ds[-1]) * 0.1,
                                lr=0.005, leakiness=0.1)

X = model['sample'](BS)
noise = np.random.randn(*X.shape) * np.sqrt(model['varx']())
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X[:, 0] + noise[:, 0], X[:, 1] + noise[:, 1])
plt.savefig('ori.png')
plt.close()
 
DATA = np.random.randn(BS, Ds[-1])
DATA /= np.linalg.norm(DATA, 2, 1, keepdims=True)

#DATA = np.linspace(-3, 3, BS)
#DATA = np.vstack([DATA, np.cos(DATA*2)]).T

#DATA = np.random.randn(BS) * 3
#DATA = np.vstack([DATA * np.cos(DATA), DATA * np.sin(DATA)]).T

DATA += np.random.randn(BS, Ds[-1]) * 0.1

DATA -= DATA.mean(0)
DATA /= DATA.max(0)
DATA *= 1

L = []
for iter in tqdm(range(250)):
    L.append(networks.EM(model, DATA, 1, 20, iter>5)[-1:])

#    print(L[-1])
#    print(L[-1][0], L[-1][-1])
#
    X = model['sample'](BS)
    noise = np.random.randn(*X.shape) * np.sqrt(model['varx']())
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(DATA[:, 0], DATA[:, 1])
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(X[:, 0] + noise[:, 0], X[:, 1] + noise[:, 1])
    plt.subplot(122)
    plt.plot(np.concatenate(L))
    plt.tight_layout()
    plt.savefig('after_{}.png'.format(sys.argv[-1]))
    plt.close()

