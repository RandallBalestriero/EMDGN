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

def create_fns(input, in_signs, Ds):
    cumulative_units = np.concatenate([[0], np.cumsum(Ds[:-1])])
    print(cumulative_units)
    Ws = [sj.initializers.he((j, i)) for i, j in zip(Ds[:-1], Ds[1:])]
    bs = [sj.initializers.he((j,)) for j in Ds[1:]]
        
    As = [T.eye(Ds[0])]
    Bs = [T.zeros(Ds[0])]
    
    As_code = [T.eye(Ds[0])]
    Bs_code = [T.zeros(Ds[0])]
    
    maps = [input]
    signs = list()
    relus = list()
    masks = [T.ones(Ds[0])]
    in_signs_ = T.concatenate([T.ones(Ds[0]), in_signs])
    
    for w, b in zip(Ws[:-1], bs[:-1]):
        
        # compute feature map and mask
        pre_activation = T.matmul(w, maps[-1])+b
        masks.append(T.where(pre_activation > 0, 1., 0.1))
        maps.append(pre_activation * masks[-1])

        # compute signs
        signs.append(T.sign(pre_activation))

    maps.append(T.matmul(Ws[-1], maps[-1]) + Bs[-1])

    for l, (w, b, m) in enumerate(zip(Ws, bs, masks)):

        As.append(T.matmul(w, As[-1] * m[:, None]))
        Bs.append(T.matmul(w, Bs[-1] * m) + b)

        start, end = cumulative_units[l], cumulative_units[l + 1]
        input_m = T.where(in_signs_[start:end] > 0, 1., 0.1)
        print(input_m, As_code[-1], w)
        As_code.append(T.matmul(w, As_code[-1] * input_m[:, None])) 
        Bs_code.append(T.matmul(w, Bs_code[-1] * input_m) + b)
    
    ineq_b = T.concatenate([b * s for b, s in zip(Bs[1:], signs)])
    ineq_A = T.vstack([a * s[:, None] for a, s in zip(As[1:], signs)])
    inequalities = T.hstack([ineq_b.reshape((-1, 1)), ineq_A])

    signs = T.concatenate(signs)

    f = sj.function(input, outputs=[maps[-1], As[-1], Bs[-1], inequalities,
                                    signs])
    g = sj.function(in_signs, outputs=[As_code[-1], Bs_code[-1]])

    return f, g 




def posterior(x, regions, g, muz, sigmaz, sigmax, ineq_A, ineq_B, init_code):
    
    alpha, m1, m2 = 0, 0, 0
    
    for region in regions:
        I, signs = region[: len(region)//2], np.array(region[len(region)//2:])
    
        new_code = np.copy(init_code)
        new_code[I] *= signs
        Aw, bw = g(new_code)
    
        mu, cov = utils.compute_mean_cov(x, Aw, bw, muz, sigmaz, sigmax)
    
        A, b = signs[:, None] * ineq_A[I], ineq_B[I]*signs
    
        if len(signs) <= len(x):
            lower, mu_, cov_, R = utils.planes_to_rectangle(A, b, mu, cov)
            invR = np.linalg.inv(R)
            m1 += invR.dot(utils.E_Y(lower, mu_, cov_))
            m2 += invR.dot(utils.E_YYT(lower, mu_, cov_).dot(invR.T))
            alpha += utils.E_1(lower - mu_, cov_)
            continue

        m = cdd.Matrix(np.hstack((b[:, None], A)))
        m.rep_type = cdd.RepType.INEQUALITY
        v = np.array(cdd.Polyhedron(m).get_generators())[:, 1:]
        for simplex in utils.get_simplices(v):
            a, s = utils.simplex_to_cones(v[simplex])
            for (A, b), ss in zip(a, s):
                lower, mu_, cov_, R = utils.planes_to_rectangle(A, b, mu, cov)
                invR = np.linalg.inv(R)
                m1 += ss * invR.dot(utils.E_Y(lower, mu_, cov_))
                m2 += ss * invR.dot(utils.E_YYT(lower, mu_, cov_).dot(invR.T))
                alpha += ss * utils.E_1(lower - mu_, cov_)
    
    return alpha, m1/alpha, m2/alpha




Ds = [2, 4, 2]
mu_z = np.zeros(Ds[0])
sigma_z = np.eye(Ds[0])
sigma_x = np.eye(Ds[-1])

input = T.Placeholder((Ds[0],), 'float32')
in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')

f, g = create_fns(input, in_signs, Ds)

x = np.random.randn(Ds[0])/10
output, A, b, inequalities, signs = f(x)
ineq_A = inequalities[:, 1:]
ineq_B = inequalities[:, 0]

regions = []
utils.search_region(ineq_A, ineq_B, regions, np.ones(len(ineq_B)))

xx = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
xx = np.vstack([xx[0].flatten(), xx[1].flatten()]).T

px = []
#pool = Pool(processes=10)
#def short(x):
#    return utils.algo2(x, ineq_A, ineq_B, signs, regions,
#            sigma_x, mu_z, sigma_z, g)[0]

#PX = pool.map(short, xx)
#print(PX)
outputs = []
for z in np.random.randn(200, 2):
    outputs.append(f(z)[0])

outputs = np.array(outputs)
plt.subplot(122)
plt.scatter(outputs[:,0], outputs[:, 1])

p = []
for x in tqdm(xx):
    p.append(utils.algo2(x, ineq_A, ineq_B, signs, regions,sigma_x, mu_z, sigma_z, g)[0])

plt.subplot(121)
plt.imshow(np.array(p).reshape(50, 50), aspect='auto')

plt.show()
print(p)
