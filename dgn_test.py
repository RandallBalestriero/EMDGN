import sys
sys.path.insert(0, "../SymJAX")
import numpy as np
import symjax as sj
import symjax.tensor as T
import cdd
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from scipy.spatial import ConvexHull
from multiprocessing import Pool


np.random.seed(14)
Ds = [1, 32,  2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])
sigma_x = np.eye(Ds[-1]) * 0.2

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

f, g, h, all_g = utils.create_fns(input, in_signs, Ds)

x = np.random.randn(Ds[0])
output, A, b, inequalities, signs = f(x)
regions = utils.search_region(all_g, g, signs)
print('number of regions:', len(regions))


N = 25
L = 2.
xx = np.meshgrid(np.linspace(-L, L, N),
                 np.linspace(-L, L, N))
xx = np.hstack([xx[0].flatten()[:, None], xx[1].flatten()[:, None]])

p = list()
for x in tqdm(xx):
    p.append(utils.algo2(x, regions, sigma_x, mu_z, sigma_z)[0])
p = np.array(p).reshape((N, N))
plt.subplot(121)
plt.imshow(np.exp(p), aspect='auto', extent=[-L, L, -L, L], origin='lower')
plt.colorbar()

plt.subplot(122)

s = list()
for z in np.random.randn(1000, Ds[0]) * np.sqrt(sigma_z[0,0]) + mu_z:
    s.append(f(z)[0])

s = np.array(s)
s += np.random.randn(len(s), Ds[-1]) * np.sqrt(sigma_x[0,0])
plt.scatter(s[:,0], s[:, 1])
plt.xlim([-L, L])
plt.ylim([-L, L])

plt.show()


asdf
plt.subplot(121)
plt.imshow(np.array(p).reshape(50, 50), aspect='auto')
plt.xlim([-L, L])
plt.ylim([-L, L])
plt.show()
print(p)
