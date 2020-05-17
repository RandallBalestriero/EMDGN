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
import datasets
import argparse
import pickle

parser = argparse.ArgumentParser(description='Compare EM and VAE.')
parser.add_argument('--dataset', type=str, help='the dataset to use',
                    choices=['circle', 'cosine'])
parser.add_argument('--leakiness', type=float, help='the activation leakiness')
parser.add_argument('--depth', type=int, help='depth of GDN')
parser.add_argument('--width', type=int, help='width of GDN')
parser.add_argument('--scale', type=float, help='scale of GDN')


args = parser.parse_args()

np.random.seed(110)

EPOCHS = 200
Ds = [1] + [args.width] * args.depth + [2]
R, BS = args.width * 4 * args.depth, 500

emt = networks.create_fns(BS, R, Ds, 1, var_x = np.ones(Ds[-1]),
                          lr=0.005, leakiness=args.leakiness)


for seed in range(10):
    
    results = []
    losses = []
    samples = []

    for sigma in [0.4, 0.1]:
    
        DATA = datasets.__dict__[args.dataset](BS, sigma)
    
        # do the VAE case
        for lr in [0.005, 0.001, 0.0001]:
            model = networks.create_vae(50, Ds, seed, lr=lr,
                                    leakiness=args.leakiness, scaler=args.scale)
            L = []
            for iter in tqdm(range(EPOCHS)):
                L.append(networks.EM(model, DATA, epochs=1, n_iter=4000)[-1])
            losses.append(L)
            emt['assign'](*model['params']())
            results.append(networks.NLL(emt, DATA))
            samples.append(model['sample'](200))
    
        # do the EM case
        model = networks.create_fns(BS, R, Ds, seed, var_x = np.ones(Ds[-1]) * 0.2,
                        var_z = np.ones(Ds[0]) * 1,
                          lr=0.005, leakiness=args.leakiness, scaler=args.scale)
        L = []
        for iter in tqdm(range(EPOCHS)):
            L.append(networks.EM(model, DATA, epochs=1, n_iter=min(5+iter,200),
                                 update_var=iter > 40)[-1])

        samples.append(model['sample'](200))
        losses.append(L)
        results.append(L[-1])
    
    
    
    filename = "saver_{}_{}_{}_{}_{}_{}.pkl".format(args.depth, args.width, args.dataset,
                                              args.leakiness, args.scale, seed)
    f = open(filename, 'wb')
    pickle.dump([losses, results, samples], f)
    f.close()

