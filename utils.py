import sys
sys.path.insert(0, "../SymJAX")
from scipy.stats import kde, multivariate_normal
import cdd
import numpy as np
import itertools
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import softmax
from numpy.linalg import lstsq
from tqdm import tqdm
import symjax as sj
import symjax.tensor as T
VERBOSE = 0


def create_fns(input, in_signs, Ds, alpha=0.1):

    cumulative_units = np.concatenate([[0], np.cumsum(Ds[:-1])])
    
    Ws = [sj.initializers.glorot((j, i)) for j, i in zip(Ds[1:], Ds[:-1])]
    bs = [sj.initializers.he((j,)) for j in Ds[1:-1]] + [T.zeros((Ds[-1],))]

    A_w = [T.eye(Ds[0])]
    B_w = [T.zeros(Ds[0])]
    
    A_q = [T.eye(Ds[0])]
    B_q = [T.zeros(Ds[0])]
    
    maps = [input]
    signs = []
    masks = [T.ones(Ds[0])]
    in_masks = T.where(T.concatenate([T.ones(Ds[0]), in_signs]) > 0, 1., alpha)

    for w, b in zip(Ws[:-1], bs[:-1]):
        
        pre_activation = T.matmul(w, maps[-1]) + b
        signs.append(T.sign(pre_activation))
        masks.append(T.where(pre_activation > 0, 1., alpha))

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

    inequalities = T.hstack([T.concatenate(B_w[1:-1])[:, None],
                             T.vstack(A_w[1:-1])]) * signs[:, None]

    inequalities_code = T.hstack([T.concatenate(B_q[1:-1])[:, None],
                                  T.vstack(A_q[1:-1])]) * in_signs[:, None]

    f = sj.function(input, outputs=[maps[-1], A_w[-1], B_w[-1],
                                    inequalities, signs])
    g = sj.function(in_signs, outputs=[A_q[-1], B_q[-1]])
    all_g = sj.function(in_signs, outputs=inequalities_code)
    h = sj.function(input, outputs=maps[-1])

    return f, g, h, all_g


def lse(x):
    x_max = x.max()
    return np.log(np.sum(np.exp(x - x_max))) + x_max


def find_region(x, regions, f):
    x_signs = []
    for x_ in x:
        x_signs.append(f(x_)[-1])
    x_signs = np.array(x_signs)

    return np.equal(x_signs[:, None, :], np.array(list(regions.keys()))).prod(2).argmax(1)


def in_region(x, ineq):
    if ineq is None:
        return np.ones(x.shape[0]).astype('bool')
    if x.ndim > 1:
        x = np.hstack([np.ones((x.shape[0], 1)), x])
    else:
        x = np.vstack([np.ones(x.shape[0]), x]).T
    return (x.dot(ineq.T) >= 0).prod(1).astype('bool')



def get_simplices(vertices):  
    """compute the simplices from a convex polytope given in its
    V-representation

    vertices: array of shape (V, D) with V the number of vertices
    """
    if vertices.shape[1] == 1:
        assert vertices.shape[0] == 2
        return [[0, 1]]
    return Delaunay(vertices).simplices


def create_H(M):
    K, D = M.shape
    A = np.copy(M)
    for i in range(D - K):
        A, b = np.vstack((A, np.random.rand(D))), np.zeros(K + 1 + i)
        b[-1] = 1
        vec = lstsq(A, b, rcond=None)[0]
        A[-1] = vec / np.linalg.norm(vec, 2)
    return A[K:]


def flip(A, i):
    sign = 1 - 2 * (np.arange(len(A)) == i).astype('float32')
    if A.ndim == 2:
        sign = sign[:, None]
    return A * sign



def find_neighbours(all_g, signs):

    ineq = all_g(np.array(signs))

    M = cdd.Matrix(np.hstack([ineq[:, [0]], ineq[:, 1:]]))
    M.rep_type = cdd.RepType.INEQUALITY
    redundant = set(M.canonicalize()[1])
    I = list(set(range(len(signs))) - redundant)
    F = np.ones((len(I), len(signs)))
    F[np.arange(len(I)), I] = -1
    return F * signs, np.array(M)



def search_region(all_g, get_Ab, signs, max_depth=9999999999999):
    S = dict()
    parents=[signs]
    for d in range(max_depth+1):
        if len(parents) == 0:
            return S
        children = []
        for s in parents:
            neighbours, M = find_neighbours(all_g, s)
            A_w, b_w = get_Ab(s)
            S[tuple(s)] = {'ineq': M, 'Ab': (A_w, b_w)}
            for n in neighbours:
                if tuple(n) not in S:
                    children.append(n)
        parents = children
    return S


def get_vertices(inequalities):
    # create the matrix the inequalities are a matrix of the form
    # [b, -A] from b-Ax>=0
    m = cdd.Matrix(inequalities)
    m.rep_type = cdd.RepType.INEQUALITY
    return cdd.Polyhedron(m).get_generators()


def mvstdnormcdf(lower, cov):
    """integrate a multivariate gaussian on rectangular domain

    Parameters
    ----------

    lower: array
        the lower bound of the rectangular region, vector of length d

    upper: array
        the upper bound of the rectangular region, vector of length d

    mu: array
        the average of the multivariate gaussian, vector of length d

    cov: array
        the covariance matrix, matrix of shape (d, d)
    """

    n = len(lower)
    upper = [np.inf] * n
    if len(cov) == 1:
        value = 1 - multivariate_normal.cdf(lower, cov=cov)
        return value
    lowinf = np.isneginf(lower)
    uppinf = np.isposinf(upper)

    infin = 2.0*np.ones(n)

    np.putmask(infin,lowinf,0)
    np.putmask(infin,uppinf,1)
    np.putmask(infin, lowinf*uppinf, -1)

    correl = cov[np.tril_indices(n, -1)]
    options = {'abseps': 1e-10, 'maxpts': 6000 * n}
    error, cdfvalue, inform = kde.mvn.mvndst(lower, upper, infin, correl,
                                             **options)

    if inform and VERBOSE:
        print('something wrong', inform, error)

    return cdfvalue


def F(x, sigma, a, mu, cov):
    value = multivariate_normal.pdf(x, cov=sigma)
    if len(cov):
        value *= mvstdnormcdf(a-mu, cov)
    return value


def Phi_0(lower, cov):
    return mvstdnormcdf(lower, cov)


def Phi1_0(lower, cov):
    """compute the first moment"""

    D = len(lower)
    f = np.zeros(D)
    valid = np.nonzero(np.isfinite(lower))[0]

    for k, low in zip(valid, lower[valid]):

        keeping = np.arange(D) != k
        cov_mk = cov[keeping, k]
        cov_ = np.delete(np.delete(cov, k, 0), k, 1)
        covk = cov_ - np.outer(cov_mk, cov_mk) / cov[k, k]

        muk = cov_mk * low / cov[k, k]

        f[k] = F(low, cov[k, k], lower[keeping], muk, covk)

    return np.matmul(cov, f)



def Phi2_0(lower, cov):
    """compute the moment 1 given a set of planes inequality
    smaller of equal to d
    """

    D = len(lower)
    f = np.zeros(D)
    ff = np.zeros((D, D))

    valid = np.nonzero(np.isfinite(lower))[0]

    for k, low in zip(valid, lower[valid]):

        keeping = np.arange(D) != k

        cov_ = np.delete(np.delete(cov, k, 0), k, 1)

        covk = cov_ - np.outer(cov[keeping, k], cov[k, keeping]) / cov[k, k]
        muk = cov[keeping, k] * low / cov[k, k]

        f[k] = F(low, cov[k, k], lower[keeping], muk, covk) * low

        for q, lowq in zip(valid, lower[valid]):
            if q == k or len(lower) <= 2:
                continue

            keeping = (np.arange(D) != k) * (np.arange(D) != q)

            cov_ = cov[keeping][:, keeping]
            cov2 = cov[keeping][:, [k, q]]
            cov22 = cov[[k, k, q, q], [k, q, k, q]].reshape((2, 2))
            inv_cov22 = np.linalg.inv(cov22)

            akq = np.array([low, lowq])

            covkq = cov_ - cov2.dot(inv_cov22.dot(cov2.T))
            mukq = cov2.dot(inv_cov22.dot(akq))

            ff[k, q] = F(akq, cov22, lower[keeping], mukq, covkq)

    first = (f + (cov * ff).sum(1)) / np.diag(cov)
    return (cov * first).dot(cov.T) + Phi_0(lower, cov) * cov



def cones_to_rectangle(ineqs, mu, cov):

    # first the general case without constraints
    if ineqs is None:
        lower = np.array([-np.inf] * len(mu))
        return lower, cov, np.eye(len(lower))

    ineqs /= np.linalg.norm(ineqs[:, 1:], 2, 1, keepdims=True)

    A, b = ineqs[:, 1:], -ineqs[:, 0]
    D = A.shape[1] - A.shape[0]
    if D == 0:
        R = A
    else:
        R = np.vstack([A, create_H(A).dot(np.linalg.inv(cov))])
    b = np.concatenate([b, np.array([-np.inf] * D)])
    l_c = b - R.dot(mu)
    cov_c = R.dot(cov.dot(R.T))
    
    return l_c, cov_c, R


def simplex_to_cones(vertices):
    S = vertices.shape[1]
    m = cdd.Matrix(np.hstack([np.ones((vertices.shape[0], 1)), vertices]))
    m.rep_type = cdd.RepType.GENERATOR
    v = np.array(cdd.Polyhedron(m).get_inequalities())

    subsets = set()
    values = set(range(v.shape[0]))
    for n in range(1, v.shape[0]):
        subsets = subsets.union(set(itertools.combinations(values, n)))

    signs = [(-1)**S] + [(-1)**(len(J) + S) for J in subsets]
    sets = [None] + [v[list(indices)] for indices in subsets]
    return sets, signs

#######################################################
#
#
#                       MU, SIGMA
#
#######################################################


def mu_sigma_all(x, A_w, b_w, sigma_z, sigma_x):

    if len(x) > 1:
        inv_sigma_x = np.linalg.inv(sigma_x)
        inv_sigma_z = np.linalg.inv(sigma_z)
    else:
        inv_sigma_x = 1/sigma_x
        inv_sigma_z = 1/sigma_z
 
    inv_sigma_w = inv_sigma_z + np.einsum('nds,dk,nkz->nsz',A_w,
                                          inv_sigma_x, A_w)
    if len(x) > 1:
        sigma_w = np.linalg.inv(inv_sigma_w)
    else:
        sigma_w = 1/inv_sigma_w
    mu_w = np.einsum('nsk,nk->ns', sigma_w,
                    np.einsum('nds,dk,nk->ns',A_w, inv_sigma_x, x - b_w))

    return mu_w, sigma_w


def mu_sigma_w(x, A_w, b_w, sigma_z, sigma_x):

    if len(x) > 1:
        inv_sigma_x = np.linalg.inv(sigma_x)
        inv_sigma_z = np.linalg.inv(sigma_z)
    else:
        inv_sigma_x = 1/sigma_x
        inv_sigma_z = 1/sigma_z
 
    inv_sigma_w = inv_sigma_z + A_w.T.dot(inv_sigma_x.dot(A_w))
    if len(x) > 1:
        sigma_w = np.linalg.inv(inv_sigma_w)
    else:
        sigma_w = 1/inv_sigma_w
    mu_w = sigma_w.dot(A_w.T.dot(inv_sigma_x.dot(x - b_w)))
    return mu_w, sigma_w


####################################################
#
#
#                   PHIS
#
#####################################################

def phis_w(ineqs, mu_w, sigma_w):

    if ineqs.shape[0] <= ineqs.shape[1]-1:
        l_c, cov_c, R_c = cones_to_rectangle(ineqs, mu_w, sigma_w)
    
        invR = np.linalg.inv(R_c)
    
        Phi_w = Phi_0(l_c, cov_c)
        Phi1_w = invR.dot(Phi1_0(l_c, cov_c))
        Phi2_w = invR.dot(Phi2_0(l_c, cov_c).dot(invR.T))\
                + np.outer(Phi1_w, mu_w) + np.outer(mu_w, Phi1_w)\
                + np.outer(mu_w, mu_w) * Phi_w

        return Phi_w, Phi1_w + mu_w * Phi_w, Phi2_w

    Phi_w = 0.
    Phi1_w = 0.
    Phi2_w = 0.

    v = np.array(get_vertices(ineqs))[:, 1:]
    for simplex in get_simplices(v):
        for ineqs_c, s in zip(*simplex_to_cones(v[simplex])):
        
            l_c, cov_c, R_c = cones_to_rectangle(ineqs_c, mu_w, sigma_w)
        
            invR = np.linalg.inv(R_c)

            Phi_w += s * Phi_0(l_c, cov_c)
            Phi1_w += s * invR.dot(Phi1_0(l_c, cov_c))
            Phi2_w += s * invR.dot(Phi2_0(l_c, cov_c).dot(invR.T))
 
    Phi2_w += np.outer(Phi1_w, mu_w) + np.outer(mu_w, Phi1_w)\
            + np.outer(mu_w, mu_w) * Phi_w
    Phi1_w += mu_w * Phi_w
    return Phi_w, Phi1_w, Phi2_w


def phis_all(ineqs, mu_all, sigma_all):
    Phi0_all = []
    Phi1_all = []
    Phi2_all = []
    for ineq, mu, sigma in zip(ineqs, mu_all, sigma_all):
        out = phis_w(ineq, mu, sigma)
        Phi0_all.append(out[0])
        Phi1_all.append(out[1])
        Phi2_all.append(out[2])
    return np.array(Phi0_all), np.array(Phi1_all), np.array(Phi2_all)


############################# kappa computations

def log_kappa_w(x, sigma_x, sigma_z, sigma_w, A_w, b_w):
    cov = sigma_x + A_w.dot(sigma_z.dot(A_w.T))
    det_w = np.linalg.det(sigma_w)
    det_s = np.linalg.det(cov)
    value = multivariate_normal.logpdf(x, mean=b_w, cov=cov)
    return 0.5*(np.log(det_w)+np.log(det_s)) + value


def log_kappa_all(x, sigma_x, sigma_z, sigma_all, A_all, b_all):
    kappas = list()
    for sigma, A, b in zip(sigma_all, A_all, b_all):
        kappas.append(log_kappa_w(x, sigma_x, sigma_z, sigma, A, b))
    return np.array(kappas)


##################################
def posterior(z, regions, x, As, Bs, mu_z, sigma_z, sigma_x):
    mus, sigmas = mu_sigma_all(x, As, Bs, sigma_z, sigma_x)
    kappas = log_kappa_all(x, sigma_x, sigma_z, sigmas, As, Bs)
    phis = phis_all([regions[r]['ineq'] for r in regions], mus, sigmas)

    w = phis[0] > 0
    c = kappas - lse(kappas[w] * np.log(phis[0][w]))
    output = np.zeros(len(z))
    for k, r in enumerate(regions):
        w = in_region(z, regions[r]['ineq'])
        output[w] = c[k] + multivariate_normal.logpdf(z[w], mean=mus[k],
                                                      cov=sigmas[k])
    return output




############################## ALGO 2
def algo2(x, regions, sigma_x, mu_z, sigma_z):

    kappas = []
    Phis_0 = []
    Phis_1 = []
    Phis_2 = []

    for flips in regions.keys():
        
        A_w, b_w = regions[flips]['Ab']

        mu_w, sigma_w = mu_sigma_w(x=x, A_w=A_w, b_w=b_w,
                                       sigma_z=sigma_z, sigma_x=sigma_x)
     
        phis = phis_w(regions[flips]['ineq'], mu_w=mu_w, sigma_w=sigma_w)

        Phis_0.append(phis[0])
        Phis_1.append(phis[1])
        Phis_2.append(phis[2])
        kappas.append(log_kappa_w(x, sigma_x=sigma_x, sigma_z=sigma_z,
                                  sigma_w=sigma_w, A_w=A_w, b_w=b_w))

 
    Phis_0 = np.array(Phis_0)
    Phis_1 = np.array(Phis_1)
    Phis_2 = np.array(Phis_2)
    kappas = np.array(kappas)

    mkappa = kappas.max()
    alphas = np.exp(kappas - mkappa) / (np.exp(kappas - mkappa)\
                                        * Phis_0).sum()

    m0_w = Phis_0 * alphas
    m1_w = Phis_1 * alphas[:, None]
    m2_w = Phis_2 * alphas[:, None, None]
    m1 = m1_w.sum(0)
    m2 = m2_w.sum(0)
    print('mean', Phis_0.mean())
    w = Phis_0 > 0
    px = lse(kappas[w] + np.log(Phis_0[w]))
    print('px', px)
    print('alpha, kappa*Phi_w, px')
    print(alphas[:3].round(2), (kappas * Phis_0)[:3].round(2),
            np.round(px,2))
    return px, m1, m2, m0_w, m1_w, m2_w



############################# evidence

