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
import networks

print('asdf')

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

Ds = [1, 16, 64]
R, BS = 110, 150
model = networks.create_fns(BS, R, Ds, 0, var_x = np.ones(Ds[-1]), lr=0.001)

DATA = datasets.load_digits(n_class=1).images
DATA = DATA[:BS].reshape((BS, -1)) + np.random.randn(BS, Ds[-1]) * 0.1
DATA /= DATA.max()
print(DATA)
L = []
for iter in tqdm(range(16)):
    L.append(networks.EM(model, DATA, 400))
    plt_state()
    plt.savefig('samples_{}.png'.format(iter))
    plt.close()

    plt.figure()
    plt.plot(np.concatenate(L), lw=3)
    plt.savefig('NLL.png')
    plt.close()


