import sys
sys.path.insert(0, "../SymJAX")
import numpy as np
import symjax as sj
import symjax.tensor as T
import cdd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import utils
from tqdm import tqdm
from scipy.spatial import ConvexHull
from multiprocessing import Pool
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


np.random.seed(6)

def create_fns(input, in_signs, Ds):

    cumulative_units = np.concatenate([[0], np.cumsum(Ds[:-1])])
    
    Ws = [sj.initializers.he((j, i)) for j, i in zip(Ds[1:], Ds[:-1])]
    bs = [sj.initializers.he((j,)) for j in Ds[1:]]

    A_w = [T.eye(Ds[0])]
    B_w = [T.zeros(Ds[0])]
    
    A_q = [T.eye(Ds[0])]
    B_q = [T.zeros(Ds[0])]
    
    maps = [input]
    signs = []
    masks = [T.ones(Ds[0])]
    in_masks = T.where(T.concatenate([T.ones(Ds[0]), in_signs]) > 0, 1.,
                                     0.1)

    for w, b in zip(Ws[:-1], bs[:-1]):
        
        pre_activation = T.matmul(w, maps[-1]) + b
        signs.append(T.sign(pre_activation))
        masks.append(T.where(pre_activation > 0, 1., 0.1))

        maps.append(pre_activation * masks[-1])

    maps.append(T.matmul(Ws[-1], maps[-1]) + bs[-1])

    # compute per region A and B
    for start, end, w, b, m in zip(cumulative_units[:-1],
                                   cumulative_units[1:], Ws, bs, masks):

        A_w.append(T.matmul(w * m, A_w[-1]))
        B_w.append(T.matmul(w * m, B_w[-1]) + b)

        A_q.append(T.matmul(w * in_masks[start:end], A_q[-1]))
        B_q.append(T.matmul(w * in_masks[start:end], B_q[-1]) + b)

    signs = T.concatenate(signs)
    ineq_b = T.concatenate(B_w[1:-1])
    ineq_A = T.vstack(A_w[1:-1])

    inequalities = T.hstack([ineq_b[:, None], ineq_A])
    inequalities = inequalities * signs[:, None] / T.linalg.norm(ineq_A, 2,
                                                         1, keepdims=True)

    inequalities_code = T.hstack([T.concatenate(B_q[1:-1])[:, None],
                                  T.vstack(A_q[1:-1])])
    inequalities_code = inequalities_code * in_signs[:, None]

    f = sj.function(input, outputs=[maps[-1], A_w[-1], B_w[-1],
                                    inequalities, signs])
    g = sj.function(in_signs, outputs=[A_q[-1], B_q[-1]])
    all_g = sj.function(in_signs, outputs=inequalities_code)
    h = sj.function(input, outputs=maps[-1])

    return f, g, h, all_g


Ds = [2, 6, 1]
input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

f, g, h, all_g = create_fns(input, in_signs, Ds)

x = np.ones(Ds[0])
output, A, b, inequalities, signs = f(x)
K = 200
X, Y = np.meshgrid(np.linspace(-2, 2, K), np.linspace(-2, 2, K))
xx = np.vstack([X.flatten(), Y.flatten()]).T


############################################################
regions = utils.search_region(all_g, g, signs)
ds = [0, 1, 2, 3, 4]

grey = matplotlib.cm.get_cmap('gray')
cmap, norm = sj.utils.create_cmap([0, 1] + list(range(2, len(ds) + 1)),
                ['w', (0.1, 0.1, 0.1)] + [grey(i) for i in np.linspace(0.3, 0.85, len(ds)-1)])
dregions = []
FINAL = np.zeros(len(xx))
fig = plt.figure(figsize=(len(ds)*4, 4.2))

for k, d in enumerate(ds):
    dregions.append(utils.search_region(all_g, g, signs, max_depth=ds[k]))
 
    ax = plt.subplot(1, len(ds) + 1, 2 + k)
    final = np.zeros(len(xx))
    if k > 0: 
        label = 'Step '+str(k)
        for sign in dregions[-1].keys()-dregions[-2].keys():
            M = dregions[-1][sign]['ineq']
            final += utils.in_region(xx, M).astype('float32') 
        FINAL += final*(k+1)
        ax.imshow(final.reshape((K, K))*(k+1), vmin=0., vmax=len(ds),
                   extent=[-2, 2, -2, 2], cmap=cmap, norm=norm, origin='lower')
        plt.xticks([])
        plt.yticks([])
    else:
        label = 'Init.'
        for M in dregions[-1]:
            final += utils.in_region(xx, dregions[-1][M]['ineq']).astype('float32')
        FINAL += final*(k+1)
        ax.imshow(final.reshape((K, K))*(k+1), vmin=0., vmax=len(ds),
                   extent=[-2, 2, -2, 2], cmap=cmap, norm=norm, origin='lower')
        plt.xticks([])
        plt.yticks([])
    for M in dregions[-1]:
        final = utils.in_region(xx, dregions[-1][M]['ineq']).astype('float32')
        ax.tricontour(xx[:,0], xx[:,1], final-0.5, levels=[0], linewidths=2)
    
    element = Line2D([0], [0], color=cmap(norm(d+1)), lw=9, label=label)
    ax.legend(handles=[element], loc='lower center', bbox_to_anchor=(0.5, -0.21),                fontsize=20)


plt.subplot(1, len(ds) + 1, 1)
plt.imshow(FINAL.reshape((K, K)), vmin=0., vmax=len(ds),
         extent=[-2, 2, -2, 2], cmap=cmap, norm=norm, origin='lower')
plt.yticks([])
plt.xticks([])

for M in regions:
    final = utils.in_region(xx, regions[M]['ineq']).astype('float32')
    plt.tricontour(xx[:,0], xx[:,1], final-0.5, levels=[0], linewidths=2)

plt.tight_layout()
plt.savefig('partition_building.pdf')
plt.close()

flips, ineq = list(regions.items())[1]
m = cdd.Matrix(np.hstack([ineq[:, [0]], ineq[:, 1:]]))
m.rep_type = cdd.RepType.INEQUALITY
v = np.array(cdd.Polyhedron(m).get_generators())[:, 1:]


####################################
simplices = utils.get_simplices(v)
plt.figure(figsize=((len(simplices) + 1)*4, 4))

plt.subplot(1, len(simplices) + 1, 1)
mask = utils.in_region(xx, ineq[:, 1:], ineq[:, 0]).astype('float32')
plt.imshow(mask.reshape((K, K)) * 2, aspect='auto', cmap=cmap, norm=norm,
           origin='lower', extent=[-2, 2, -2, 2])
plt.xticks([])
plt.yticks([])

for M in list(regions.values()):
    final = utils.in_region(xx, M[:, 1:], M[:, 0]).astype('float32')
    plt.tricontour(xx[:,0], xx[:,1], final-0.5, levels=[0], linewidths=2)


for i, simplex in enumerate(simplices):
    A_w, b_w = g(np.array(flips))

    m = cdd.Matrix(np.hstack([np.ones((len(simplex),1)),v[simplex]]))
    m.rep_type = cdd.RepType.GENERATOR
    m = np.array(cdd.Polyhedron(m).get_inequalities())

    plt.subplot(1, len(simplices) + 1, 2 + i)
    mask = utils.in_region(xx, m[:, 1:], m[:, 0]).astype('float32')
    plt.imshow(mask.reshape((K, K)), aspect='auto', cmap=cmap, norm=norm,
               origin='lower', extent=[-2, 2, -2, 2])
    for M in list(regions.values()):
        final = utils.in_region(xx, M[:, 1:], M[:, 0]).astype('float32')
        plt.tricontour(xx[:,0], xx[:,1], final-0.5, levels=[0], linewidths=2)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig('region_to_simplices.pdf')
plt.close()

####################################3

final = np.zeros(xx.shape[0])
simplex = simplices[0]
a, signs = utils.simplex_to_cones(v[simplex])
fig = plt.figure(figsize=((len(a)+0.8)*4, 8))
k = 1
for (A, b), sign in zip(a, signs):
    plt.subplot(2, len(a), k)
    yy = utils.in_region(xx, A, b).astype('float32')
    final += sign * yy
    plt.imshow(yy.reshape((K, K)), vmin=0, vmax=4, cmap='Greys',
            origin='lower', aspect='auto')
    plt.title(str(sign), fontsize=25)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, len(a), len(a) + k)
    plt.imshow(final.reshape((K, K)), vmin=-4, vmax=4, cmap='RdGy',
            origin='lower', aspect='auto')
    plt.xticks([])
    plt.yticks([])

    k += 1

elements = []
cmap = matplotlib.cm.get_cmap('RdGy')
for v in [-1, 0, 1, 2]:
    elements.append(Patch(facecolor=cmap((v+4)/8), edgecolor='k', label=str(v)))

fig.legend(handles=elements, loc='right', fontsize=30, ncol=1,
           bbox_to_anchor=(1.002, 0.5))

plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.savefig('simplex_to_cones.pdf')

