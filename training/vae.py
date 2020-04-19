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

Ds = [1, 14, 2]
R, BS = 80, 200

#model = networks.create_vae(BS, Ds, 0, lr=0.01)
model = networks.create_fns(BS, R, Ds, 1, log_sigma_x = np.log(0.2 + np.zeros(Ds[-1])),
                            lr=0.005, leakiness=0.001)


DATA = np.random.randn(BS, Ds[-1])
DATA /= np.linalg.norm(DATA, 2, 1, keepdims=True)

#DATA = np.linspace(-3, 3, BS)
#DATA = np.vstack([DATA, np.cos(DATA*2)]).T
DATA += np.random.randn(BS, Ds[-1]) * 0.1

L = []
for iter in tqdm(range(350)):
    L.append(networks.EM(model, DATA, 100))
    print(L[-1][0], L[-1][-1])

    X = model['sample'](200)
    plt.subplot(121)
    plt.scatter(DATA[:, 0], DATA[:, 1])
    plt.scatter(X[:, 0], X[:, 1])
    plt.subplot(122)
    plt.plot(np.concatenate(L))
    plt.savefig('after.png')
    plt.close()
    
