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
from sklearn import datasets
import networks

def plt_state():
    predictions = model['sample'](100)
    
    plt.figure(figsize=(30, 30))
    for i in range(50):
        plt.subplot(10, 10, 1 + i)
        plt.imshow(predictions[i].reshape((8, 8)))
    for i in range(50):
        plt.subplot(10, 10, 51 + i)
        plt.imshow(DATA[i].reshape((8, 8)))





np.random.seed(int(sys.argv[-1]) + 10)

Ds = [1, 12, 64]
R, BS = 110, 150

sigma_x = .1
model = networks.create_fns(BS, R, Ds, 1, lr=0.001, leakiness=0.1, var_x=sigma_x**2)

DATA = datasets.load_digits(n_class=1).images
DATA = DATA[:BS].reshape((BS, -1)) + np.random.randn(BS, Ds[-1]) * 0.1
#DATA -= DATA.mean(1, keepdims=True)
DATA /= (DATA.max() * 10)
print(DATA)
L = []

for iter in tqdm(range(36)):

    plt_state()
    plt.savefig('samples_{}.png'.format(iter))
    plt.close()

    L.append(networks.EM(model, DATA, 1, 10, pretrain=iter==0, update_var=iter>5))
    print(L[-1])

    plt.figure()
    plt.plot(np.concatenate(L), lw=3)
    plt.savefig('NLL.png')
    plt.close()


