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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--std', type=float, default=0.55)#0.15 # 0.4 is good for mnsit
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--network', type=str)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--leakiness', type=float, default=0.1)
parser.add_argument('--noise', type=float, default=0.1)
args = parser.parse_args()

np.random.seed(args.seed)
if args.dataset == 'mnist':
    BS = 1000
else:
    BS = 500

DATA = networks.create_dataset(args.dataset, BS, noise_std=args.noise)
if args.network == 'small':
    Ds = [1, 8, DATA.shape[1]]
    R = 16
elif args.network == 'large':
    Ds = [1, 8, 8, DATA.shape[1]]
    R = 64

elif args.network == 'xlarge':
    Ds = [1, 32, 32, DATA.shape[1]]
    R = 128




graph = sj.Graph('test')
with graph:
    if args.model != 'EM':
        lr = sj.tensor.Variable(1., name='lr', trainable=False)
        emt = networks.create_fns(BS, R, Ds, 1, var_x = np.ones(Ds[-1]) * args.std**2,
                          lr=0.005, leakiness=args.leakiness)
        model = networks.create_vae(50, Ds, args.seed, lr=lr,
                                leakiness=args.leakiness, scaler=1)
    else:
        model = networks.create_fns(BS, R, Ds, 1, lr=0.001, leakiness=args.leakiness,
                                    var_x=args.std**2)
 

for RUN in range(20):
    # do the VAE case
    if args.model != 'EM':
        filename = 'nnnnsaving_likelihood_{}_{}_{}_{}_{}_{}.npz'
        for lr_ in [0.005, 0.001, 0.0001]:
            graph.reset()
            lr.update(lr_)
            out = networks.EM(model, DATA, epochs=args.epochs, n_iter=500, extra=emt)
            np.savez(filename.format(args.dataset, args.epochs, args.model,
                                    lr_, args.network, RUN), L=out[0], LL=out[1],
                                    samples=model['sample'](4*BS),
                                    noise=np.random.randn(4*BS,2) * np.sqrt(model['varx']()), data=DATA)
    else:
        graph.reset()
        filename = 'nnnsaving_likelihood_{}_{}_{}_{}_{}.npz'
        out = networks.EM(model, DATA, epochs=args.epochs, n_iter=20, update_var=2)
        np.savez(filename.format(args.dataset, args.epochs, args.model, args.network, RUN), L=out,
                    samples=model['sample'](4*BS), noise=np.random.randn(4*BS,Ds[-1]) * np.sqrt(model['varx']()), data=DATA)

