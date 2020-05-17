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
from multiprocessing import Pool, Array

import networks

VERBOSE = 0


def create_fns(input, in_signs, Ds, x, m0, m1, m2, batch_in_signs, alpha=0.1,
                sigma=1, sigma_x=1, lr=0.0002):

    cumulative_units = np.concatenate([[0], np.cumsum(Ds[:-1])])
    BS = batch_in_signs.shape[0]
    Ws = [T.Variable(sj.initializers.glorot((j, i)) * sigma)
                    for j, i in zip(Ds[1:], Ds[:-1])]
    bs = [T.Variable(sj.initializers.he((j,)) * sigma) for j in Ds[1:-1]]\
                + [T.Variable(T.zeros((Ds[-1],)))]

    A_w = [T.eye(Ds[0])]
    B_w = [T.zeros(Ds[0])]
    
    A_q = [T.eye(Ds[0])]
    B_q = [T.zeros(Ds[0])]
    
    batch_A_q = [T.eye(Ds[0]) * T.ones((BS, 1, 1))]
    batch_B_q = [T.zeros((BS, Ds[0]))]
    
    maps = [input]
    signs = []
    masks = [T.ones(Ds[0])]

    in_masks = T.where(T.concatenate([T.ones(Ds[0]), in_signs]) > 0, 1., alpha)
    batch_in_masks = T.where(T.concatenate([T.ones((BS, Ds[0])),
                                            batch_in_signs], 1) > 0, 1., alpha)

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

        batch_A_q.append(T.matmul(w * batch_in_masks[:, None, start:end],
                                  batch_A_q[-1]))
        batch_B_q.append((w * batch_in_masks[:, None, start:end]\
                            * batch_B_q[-1][:, None, :]).sum(2) + b)

    batch_B_q = batch_B_q[-1]
    batch_A_q = batch_A_q[-1]

    signs = T.concatenate(signs)

    inequalities = T.hstack([T.concatenate(B_w[1:-1])[:, None],
                             T.vstack(A_w[1:-1])]) * signs[:, None]

    inequalities_code = T.hstack([T.concatenate(B_q[1:-1])[:, None],
                                  T.vstack(A_q[1:-1])]) * in_signs[:, None]

    #### loss
    log_sigma2 = T.Variable(sigma_x)
    sigma2 = T.exp(log_sigma2)

    Am1 = T.einsum('qds,nqs->nqd', batch_A_q, m1)
    Bm0 = T.einsum('qd,nq->nd', batch_B_q, m0)
    B2m0 = T.einsum('nq,qd->n', m0, batch_B_q**2)
    AAm2 = T.einsum('qds,qdu,nqup->nsp', batch_A_q, batch_A_q, m2)
    
    inner = - (x * (Am1.sum(1) + Bm0)).sum(1) + (Am1 * batch_B_q).sum((1, 2))

    loss_2 = (x ** 2).sum(1) + B2m0 + T.trace(AAm2, axis1=1, axis2=2).squeeze()

    loss_z = T.trace(m2.sum(1), axis1=1, axis2=2).squeeze()

    cst = 0.5 * (Ds[0] + Ds[-1]) * T.log(2 * np.pi)

    loss = cst + 0.5 * Ds[-1] * log_sigma2 + inner / sigma2\
            + 0.5 * loss_2 / sigma2 + 0.5 * loss_z

    mean_loss = loss.mean()
    adam = sj.optimizers.NesterovMomentum(mean_loss, Ws + bs, lr, 0.9)

    train_f = sj.function(batch_in_signs, x, m0, m1, m2, outputs=mean_loss, updates=adam.updates)
    f = sj.function(input, outputs=[maps[-1], A_w[-1], B_w[-1],
                                    inequalities, signs])
    g = sj.function(in_signs, outputs=[A_q[-1], B_q[-1]])
    all_g = sj.function(in_signs, outputs=inequalities_code)
    h = sj.function(input, outputs=maps[-1])
    return f, g, h, all_g, train_f, sigma2


def lse(x, axis):
    x_max = x.max(axis=axis)
    return np.log(np.sum(np.exp(x - x_max[:, None]), axis=axis)) + x_max


def find_region(z, regions, input2sign):
    x_signs = np.array([input2sign(zz.reshape((-1,))) for zz in z])
    signs = np.array(list(regions.keys()))
    return np.equal(x_signs[:, None, :], signs).all(2).argmax(1)


def in_region(z, ineq):
    """
    z is shape (N, S) or (S)
    ineq is shape (K, S+1)
    """
    
    if z.ndim > 1:
        if ineq is None:
            return np.ones(z.shape[0], dtype='bool')
        zp = np.hstack([np.ones((z.shape[0], 1)), z])
        return (np.einsum('ks,ns->nk', ineq, zp) >= 0).all(axis=1)
    else:
        if ineq is None:
            return True
        return (ineq.dot(np.hstack([np.ones(1), z])) >= 0).all()



def get_simplices(vertices):  
    """compute the simplices from a convex polytope given in its
    V-representation

    vertices: array of shape (V, D) with V the number of vertices
    """
    if vertices.shape[1] == 1:
        assert vertices.shape[0] == 2
        return [[0, 1]]
    return Delaunay(vertices).simplices

def flip(A, i):
    sign = 1 - 2 * (np.arange(len(A)) == i).astype('float32')
    if A.ndim == 2:
        sign = sign[:, None]
    return A * sign

def reduce_ineq(ineqs):
    norms = set(np.nonzero(np.linalg.norm(ineqs, 2, 1) < 1e-8)[0])
    M = cdd.Matrix(ineqs)
    M.rep_type = cdd.RepType.INEQUALITY

    I = list(set(range(len(ineqs))) - norms - set(M.canonicalize()[1]))
    return I

 


def find_neighbours(signs2ineq, signs):

    ineq = signs2ineq(np.array(signs))
    I = reduce_ineq(ineq)

    # create the sign switching table
    F = np.ones((len(I), len(signs)))
    F[np.arange(len(I)), I] = - 1

    return F * signs


def search_region(signs2ineq, signs2Ab, signs, max_depth=9999999999999):
    S = dict()
    # init
    all_signs = []
    # init the to_visit
    to_visit=[list(signs)]
    # search all the regions (their signs)
    while True:
        all_signs += to_visit
        to_visit_after = []
        for s in to_visit:
            neighbours = find_neighbours(signs2ineq, s)
            for n in neighbours:
                a = np.any([np.array_equal(n,p) for p in to_visit_after])
                b = np.any([np.array_equal(n,p) for p in to_visit])
                c = np.any([np.array_equal(n,p) for p in all_signs])
                if not (a + b + c):
                    to_visit_after.append(n)
        if len(to_visit_after) == 0:
            break
        to_visit = to_visit_after
    # not set up S
    for s in all_signs:
        ineq = signs2ineq(s)
        S[tuple(s)] = {'ineq': ineq[reduce_ineq(ineq)], 'Ab': signs2Ab(s)}
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
    
    if n == 1:
        return 1 - multivariate_normal.cdf(lower, cov=cov)

    upper = [np.inf] * n
    lowinf = np.isneginf(lower)
    uppinf = np.isposinf(upper)

    infin = 2.0 * np.ones(n)

    np.putmask(infin,lowinf,0)
    np.putmask(infin,uppinf,1)
    np.putmask(infin, lowinf*uppinf, -1)

    correl = cov[np.tril_indices(n, -1)]
    options = {'abseps': 1e-20, 'maxpts': 6000 * n}
    error, cdfvalue, inform = kde.mvn.mvndst(lower, upper, infin, correl,
                                             **options)

    if inform:
        print('something wrong', inform, error)

    return cdfvalue




def mu_u_sigma_u(low, cov, u):

    D = len(cov)

    if np.isscalar(u):
        keeping = np.arange(D) != u
    else:
        keeping = (np.arange(D) != u[0]) * (np.arange(D) != u[1])

    cov_no_u = cov[keeping][:, u]
    cov_u = cov[keeping][:, keeping]
    if np.isscalar(u):
        cov_ = cov_u - np.outer(cov_no_u, cov_no_u) / cov_u
        mu_ = cov_no_u * low[u] / cov_u
    else:
        inv_cov_u = np.linalg.inv(cov_u)
        cov_ = cov_u - cov_no_u.dot(inv_cov_u.dot(cov_no_u.T))
        mu_ = cov_no_u.dot(inv_cov_u.dot(low[u]))
    return mu_, cov_, low[keeping]


def get_F_G(lower, cov):
    """compute the moment 1 given a set of planes inequality
    smaller of equal to d
    """

    D = len(lower)
    f = np.zeros(D)
    g = np.zeros((D, D))

    for k in range(len(lower)):

        if lower[k] == - np.inf:
            continue

        f[k] = multivariate_normal.pdf(lower[k], cov=cov[k, k])
        if len(cov) > 1:
            mu_u, cov_u, low_no_u = mu_u_sigma_u(lower, cov, k)
            f[k] *= mvstdnormcdf(low_no_u - mu_u, cov_u)

        for q in range(len(lower)):

            if q == k or len(lower) <= 2 or lower[q] == - np.inf:
                continue
    
            u = [k, q]
            g[k, q] = multivariate_normal.pdf(lower[u], cov=cov[u][:, u])

            if len(cov) > 2:
                mu_u, cov_u, low_no_u = mu_u_sigma_u(lower, cov, u)
                g[k, q] *= mvstdnormcdf(low_no_u - mu_u, cov_u)

    return f, g



def create_H(M):
    K, D = M.shape
    A = np.copy(M)
    for i in range(D - K):
        A, b = np.vstack((A, np.ones((1, D)))), np.zeros(K + 1 + i)
        b[-1] = 1
        vec = lstsq(A, b, rcond=-1)[0]
        A[-1] = vec / np.linalg.norm(vec, 2)
    return A[K:]



def cones_to_rectangle(ineqs, cov):

    # first the general case without constraints
    if ineqs is None:
        lower = np.array([-np.inf] * len(cov))
        return lower, np.eye(len(lower))

    ineqs /= np.linalg.norm(ineqs[:, 1:], 2, 1, keepdims=True)

    A, b = ineqs[:, 1:], - ineqs[:, 0]
    D = A.shape[1] - A.shape[0]
    if D == 0:
        R = A
    else:
        R = np.vstack([A, create_H(A)])
        R = np.vstack([A, create_H(A).dot(np.linalg.inv(cov))])
    return np.concatenate([b, np.array([-np.inf] * D)]), R

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


def mu_sigma(x, A, b, cov_z, cov_x):
    """takes a matrix of data x, all the region A and b and the cov x and x
    returns n covariance matrices and N x n bias vectors
    """

    inv_cov_x = np.linalg.inv(cov_x) if A.shape[1] > 1 else 1/cov_x
    inv_cov_z = np.linalg.inv(cov_z) if A.shape[2] > 1 else 1/cov_z

    inv_cov_w = inv_cov_z + np.einsum('nds,dk,nkz->nsz',A, inv_cov_x, A)

    cov_w = np.linalg.inv(inv_cov_w) if A.shape[2] > 1 else 1/inv_cov_w
    
    mu_w = np.einsum('nsk,Nnk->Nns', cov_w, np.einsum('nds,dk,Nnk->Nns',
                                    A, inv_cov_x, x[:, None, :] - b))
    
    return mu_w, cov_w


####################################################
#
#                   PHIS
#
####################################################

def phis_w(ineq_w, mu, cov_w):

    # instead of integrating a non centered gaussian on w 
    # we integrate a centered Gaussian on w-mu. This is equivalent to 
    # adding mu to the bias of the inequality system
    ineqs = ineq_w + 0.
    ineqs[:, 0] += ineqs[:, 1:].dot(mu)

    # we initialize the accumulators
    phi0, phi1, phi2 = 0., 0., 0.
    print(ineqs / np.linalg.norm(ineqs[:, 1:], 2, 1, keepdims=True))
    if ineqs.shape[0] <= ineqs.shape[1] - 1:
        simplices = [range(len(ineqs))]
    else:
        v = np.array(get_vertices(ineqs))[:, 1:]
        print(v)
        simplices = get_simplices(v)

    for simplex in simplices:

        cones = [(ineqs, 1)] if ineqs.shape[0] <= ineqs.shape[1] - 1 else zip(*simplex_to_cones(v[simplex]))

        for ineqs_c, s in cones:

            l_c, R_c = cones_to_rectangle(ineqs_c, cov_w)
            cov_c = R_c.dot(cov_w.dot(R_c.T))
            f, G = get_F_G(l_c, cov_c)
    
            phi0 += s * mvstdnormcdf(l_c, cov_c)
            phi1 += s * R_c.T.dot(f) # THIS SHOULD BE CHANGED BELOW FOR S>1
            H = np.diag(np.nan_to_num(l_c) * f / np.diag(cov_c))############ - (cov_c * G).sum(1)) / np.diag(cov_c))
            phi2 += s * R_c.T.dot(H.dot(R_c))

    phi1 = cov_w.dot(phi1)
    phi2 = (cov_w + np.outer(mu, mu))* phi0 + cov_w.dot(phi2.dot(cov_w))\
            + np.outer(mu, phi1) + np.outer(phi1, mu)
    phi1 += mu * phi0

    return phi0, phi1, phi2


def phis_all(ineqs, mu_all, cov_all):

    phi0 = np.zeros(len(ineqs))
    phi1 = np.zeros((len(ineqs), cov_all.shape[-1]))
    phi2 = np.zeros(phi1.shape + (cov_all.shape[-1],))

    for i, (ineq, mu, cov) in enumerate(zip(ineqs, mu_all, cov_all)):
        phi0[i], phi1[i], phi2[i] = phis_w(ineq, mu, cov)

    return phi0, phi1, phi2


############################# kappa computations

def log_kappa(x, cov_x, cov_z, A, b):
    cov = cov_x + np.einsum('nds,sp,nkp->ndk',A, cov_z, A)
    kappas = np.array([multivariate_normal.logpdf(x, mean=m, cov=c)
                       for m, c in zip(b, cov)])
    if x.shape[0] == 1:
        return kappas[None, :]
    return kappas.T


##################################
def posterior(z, regions, x, As, Bs, cov_z, cov_x, input2signs):
    mu, cov = mu_sigma(x, As, Bs, cov_z, cov_x)
    
    kappas = np.exp(log_kappa(x, cov_x, cov_z, As, Bs))[0]
    
    phis0 = phis_all([regions[r]['ineq'] for r in regions], mu[0], cov)[0]

    indices = find_region(z, regions, input2signs)
    output = np.zeros(len(indices))
    for k in np.unique(indices):
        w = indices == k
        output[w] = multivariate_normal.pdf(z[w], mean=mu[0, k], cov=cov[k])
    output *= kappas[indices] / (kappas * phis0).sum()
    return output




############################## ALGO 2
def marginal_moments(x, regions, cov_x, cov_z):

    # find all regions of the DN
    As = np.array([regions[s]['Ab'][0] for s in regions])
    Bs = np.array([regions[s]['Ab'][1] for s in regions])

    # find all mus and cov (cov is constant per x, not mus)
    mus, covs = mu_sigma(x, As, Bs, cov_z, cov_x) #(N n D) (n D D)
    log_kappas = log_kappa(x, cov_x, cov_z, As, Bs) #(N n)
    
    ineqs = np.array([regions[r]['ineq'] for r in regions])
    
    P0, P1, P2 = [], [], []
    for mu in tqdm(mus, desc='Computing PHIS'):
        p0, p1, p2 = phis_all(ineqs, mu, covs)
        P0.append(p0)
        P1.append(p1)
        P2.append(p2)
    phis = [np.array(P0), np.array(P1), np.array(P2)]

    phis[0] = np.maximum(phis[0], 1e-30)
    phis[2] = np.maximum(phis[2], 1e-30 * np.eye(len(cov_z)))

    # compute marginal
    px = np.exp(lse(log_kappas + np.log(phis[0]), axis=1)) # (N)

    # compute per region moments
    print(mus.shape, log_kappas.shape)
    alphas = np.exp(log_kappas - log_kappas.max(1, keepdims=True))\
            / (np.exp(log_kappas - log_kappas.max(1, keepdims=True)) * phis[0]).sum(1, keepdims=True)

    m0_w = phis[0] * alphas
    m1_w = phis[1] * alphas[:, :, None]
    m2_w = phis[2] * alphas[:, :, None, None]

    return px, m0_w, m1_w, m2_w
