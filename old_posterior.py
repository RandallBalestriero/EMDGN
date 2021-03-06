import sys
sys.path.insert(0, "../../SymJAX")
sys.path.insert(0, "../")
import numpy as np
import symjax as sj
import symjax.tensor as T
import cdd
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from scipy.spatial import ConvexHull
from multiprocessing import Pool
import matplotlib

np.random.seed(int(sys.argv[-1]) + 10)

Ds = [1, 16, 2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0]) * 2
sigma_x = np.eye(Ds[-1]) * 0.05

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

R, BS = 40, 50

batch_in_signs = T.Placeholder((R, np.sum(Ds[1:-1])), 'bool')
x = T.Placeholder((BS, Ds[-1]), 'float32')
m0 = T.Placeholder((BS, R), 'float32')
m1 = T.Placeholder((BS, R, Ds[0]), 'float32')
m2 = T.Placeholder((BS, R, Ds[0], Ds[0]), 'float32')

f, g, h, all_g, train_f = utils.create_fns(input, in_signs, Ds, x, m0, m1, m2, batch_in_signs)



x = np.random.randn(Ds[0])/4
output, A, b, inequalities, signs = f(x)

regions = utils.search_region(all_g, g, signs)
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

C = 4

cmap = matplotlib.cm.get_cmap('Spectral')
plt.figure(figsize=(16,4))
for i in range(C):
    z = np.random.randn(Ds[0])
    output, A, b, inequalities, signs = f(z)
    output = output + np.random.randn(2) * 0.1
    p = utils.posterior(xx, regions, output, As, Bs, mu_z, sigma_z, sigma_x)
    print(p)
    plt.subplot(1, 4, 1)
    if Ds[0] == 1:
        plt.plot(xx, p)
        plt.axvline(z, color='k')
        for x,t in zip(xxflag, np.linspace(0, 1, len(xxflag))):
            plt.axvline(x, color=cmap(t))
    else:
        plt.imshow((np.array(p)).reshape((N, N)), aspect='auto',
                    extent=[-L, L, -L, L], origin='lower')
        plt.colorbar()
    plt.title(r'$p(z|x)$')
    plt.subplot(1, 4, 2)
    predictions = list()
    for z in xx.reshape((-1, 1)):
        predictions.append(f(z)[0])
    predictions = np.array(predictions)
    plt.scatter(predictions[:,0], predictions[:, 1])

    for z, t in zip(xxflag.reshape((-1, 1)), np.linspace(0, 1, len(xxflag))):
        prediction = f(z)[0]
        plt.scatter(prediction[0], prediction[1], color=cmap(t))
    plt.scatter(output[0], output[1], color='k')

    N = 35
    X0, X1 = predictions[:, 0].min(), predictions[:, 0].max()
    Y0, Y1 = predictions[:, 1].min(), predictions[:, 1].max()
    X0 -= 0.5
    X1 += 0.5
    Y0 -= 0.5
    Y1 += 0.5
    xxx = np.meshgrid(np.linspace(X0, X1, N), np.linspace(Y0, Y1, N))
    xxx = np.hstack([xxx[0].flatten()[:, None], xxx[1].flatten()[:, None]])
    
    p = list()
    for x in tqdm(xxx):
        p.append(utils.algo2(x, regions, sigma_x, mu_z, sigma_z)[0])
    p = np.array(p).reshape((N, N))
    plt.title(r'$g(z)$')

    plt.subplot(143)
    noise = np.random.randn(*predictions.shape)*np.sqrt(sigma_x[0,0])
    plt.scatter(predictions[:,0] + noise[:, 0],
                predictions[:, 1] + noise[:, 1], color='blue')
    plt.scatter(predictions[:,0], predictions[:, 1], color='red')
    plt.title(r'$g(z)$ and $g(z)+\epsilon$')

    plt.subplot(144)
    plt.imshow(np.exp(p), aspect='auto', extent=[X0, X1, Y0, Y1], origin='lower')
    plt.title(r'$p(x)$')
    #plt.scatter(predictions[:,0], predictions[:, 1])


    plt.savefig('test1_{}.png'.format(sys.argv[-1]))
    asdf
#    for k, ss in enumerate(regions):
#        PRED += regions[ss]['Ab'][0].dot(m1_w[k])+regions[ss]['Ab'][1]*m0_w[k]
#        final = utils.in_region(xx, regions[ss]['ineq']).astype('float32')
#        plt.tricontour(xx[:,0], xx[:,1], final-0.5, levels=[0], linewidths=2)
#        plt.scatter([mus[k,0]], [mus[k,1]], c='r')
#
#    print('pred', PRED, output)
#    s = np.array([x])
#    plt.scatter(s[:,0], s[:, 1])
#    plt.xlim([-L, L])
#    plt.ylim([-L, L])
#    plt.subplot(2, C, C + 1 + i)
#    plt.scatter([output[0]], [output[1]])
#    plt.scatter([PRED[0]], [PRED[1]], c='r')
#    plt.xlim([-L, L])
#    plt.ylim([-L, L])
#plt.show()
#
#
#asdf
#plt.subplot(121)
#plt.imshow(np.array(p).reshape(50, 50), aspect='auto')
#plt.xlim([-L, L])
#plt.ylim([-L, L])
#plt.show()
#print(p)
