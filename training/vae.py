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



np.random.seed(int(sys.argv[-1]) + 10)

Ds = [1, 8, 2]
R, BS = 80, 200

#model = networks.create_vae(BS, Ds, 1, lr=0.01)
model = networks.create_fns(BS, R, Ds, 1, var_x = (0.3 **2) * np.ones(Ds[-1]),
                            lr=0.001, leakiness=0.01)

X = model['sample'](200)
noise = np.random.randn(*X.shape) * np.sqrt(model['varx']())
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X[:, 0] + noise[:, 0], X[:, 1] + noise[:, 1])
plt.savefig('ori.png')
plt.close()
 
DATA = np.random.randn(BS, Ds[-1])
DATA /= np.linalg.norm(DATA, 2, 1, keepdims=True)

DATA = np.linspace(-3, 3, BS)
DATA = np.vstack([DATA, np.cos(DATA*2)]).T
DATA += np.random.randn(BS, Ds[-1]) * np.sqrt(0.05)

L = []
for iter in tqdm(range(150)):
    L.append(networks.EM(model, DATA, 300))
    print(L[-1])

    X = model['sample'](200)
    print('asdf', np.sqrt(model['varx']()))
    noise = np.random.randn(*X.shape) * np.sqrt(model['varx']())
    plt.subplot(121)
    plt.scatter(DATA[:, 0], DATA[:, 1])
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(X[:, 0] + noise[:, 0], X[:, 1] + noise[:, 1])
    plt.subplot(122)
    plt.plot(np.concatenate(L))
    plt.savefig('after.png')
    plt.close()
    
