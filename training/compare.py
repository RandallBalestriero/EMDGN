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


np.random.seed(110)

Ds = [1, 6, 2]
R, BS = 100, 250

vae = networks.create_vae(50, Ds, 1, lr=0.0005, leakiness=0.01, scaler=3)
emt = networks.create_fns(BS, R, Ds, 1, var_x = np.ones(Ds[-1]),
                                lr=0.005, leakiness=0.1)
em = networks.create_fns(BS, R, Ds, 1, var_x = np.ones(Ds[-1]),
                                lr=0.005, leakiness=0.1)

DATA = np.random.randn(BS, Ds[-1])
DATA /= np.linalg.norm(DATA, 2, 1, keepdims=True)

#DATA = np.linspace(-3, 3, BS)
#DATA = np.vstack([DATA, np.cos(DATA*2)]).T

#DATA = np.random.randn(BS) * 3
#DATA = np.vstack([DATA * np.cos(DATA), DATA * np.sin(DATA)]).T

DATA += np.random.randn(BS, Ds[-1]) * 0.1

DATA -= DATA.mean(0)
DATA /= DATA.max(0)
DATA *= 3

Q = []
L = []
TL = []
for iter in tqdm(range(4000)):
    L.append(networks.EM(vae, DATA, epochs=1, n_iter=1000)[-1])
#    Q.append(networks.EM(em, DATA, epochs=1, n_iter=20, update_var=iter>8)[-1])
    emt['assign'](*vae['params']())
    TL.append(networks.NLL(emt, DATA))


plt.plot(np.linspace(0, 1, len(L)), L)
plt.plot(np.linspace(0, 1, len(Q)), Q)
plt.plot(np.linspace(0, 1, len(TL)), TL)

#    print(L[-1])
#    print(L[-1][0], L[-1][-1])
#
plt.figure(figsize=(12, 6))
for k, model in enumerate([vae, em]):
    X = model['sample'](BS)
    noise = np.random.randn(*X.shape) * np.sqrt(model['varx']()) * 0
    plt.subplot(1,2, k + 1)
    plt.scatter(DATA[:, 0], DATA[:, 1], c='k')
    plt.scatter(X[:, 0] + noise[:, 0], X[:, 1] + noise[:, 1], c='b')

plt.show()
    
