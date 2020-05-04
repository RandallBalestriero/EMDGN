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


def lse(x):
    x_max = x.max()
    return np.log(np.sum(np.exp(x - x_max))) + x_max


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


def create_H(M):
    K, D = M.shape
    A = np.copy(M)
    for i in range(D - K):
        A, b = np.vstack((A, np.random.randn(1, D))), np.zeros(K + 1 + i)
        b[-1] = 1
        vec = lstsq(A, b, rcond=None)[0]
        A[-1] = vec / np.linalg.norm(vec, 2)
    return A[K:]


def flip(A, i):
    sign = 1 - 2 * (np.arange(len(A)) == i).astype('float32')
    if A.ndim == 2:
        sign = sign[:, None]
    return A * sign



def find_neighbours(signs2ineq, signs):

    ineq = signs2ineq(np.array(signs))

    M = cdd.Matrix(ineq)
    M.rep_type = cdd.RepType.INEQUALITY
    redundant = set(M.canonicalize()[1])
    I = list(set(range(len(signs))) - redundant)
    F = np.ones((len(I), len(signs)))
    F[np.arange(len(I)), I] = - 1
    return F * signs, np.array(M)



def search_region(signs2ineq, signs2Ab, signs, max_depth=9999999999999):
    S = dict()
    parents=[signs]
    for d in range(max_depth+1):
        if len(parents) == 0:
            return S
        children = []
        for s in parents:
            neighbours, M = find_neighbours(signs2ineq, s)
            A_w, b_w = signs2Ab(s)
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



def get_F_G(lower, cov):
    """compute the moment 1 given a set of planes inequality
    smaller of equal to d
    """

    D = len(lower)
    f = np.zeros(D)
    g = np.zeros((D, D))

    valid = np.nonzero(np.isfinite(lower))[0]

    for k, low in zip(valid, lower[valid]):

        keeping = np.arange(D) != k

        cov_ = np.delete(np.delete(cov, k, 0), k, 1)

        covk = cov_ - np.outer(cov[keeping, k], cov[k, keeping]) / cov[k, k]
        muk = cov[keeping, k] * low / cov[k, k]

        f[k] = F(low, cov[k, k], lower[keeping], muk, covk)# * low

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

            g[k, q] = F(akq, cov22, lower[keeping], mukq, covkq)

    return f, g



def cones_to_rectangle(ineqs, cov):

    # first the general case without constraints
    if ineqs is None:
        lower = np.array([-np.inf] * len(cov))
        return lower, cov, np.eye(len(lower))

#    l2 = np.linalg.norm(ineqs[:, 1:], 2, 1)
#    ineqs = ineqs[l2 > 0]
#    ineqs /= np.linalg.norm(ineqs[:, 1:], 2, 1, keepdims=True)

    A, b = ineqs[:, 1:], - ineqs[:, 0]
    D = A.shape[1] - A.shape[0]
    if D == 0:
        R = A
    else:
        R = np.vstack([A, create_H(A)])
#        R = np.vstack([A, create_H(A).dot(np.linalg.inv(cov))])
    cov_c = R.dot(cov.dot(R.T))
    b = np.concatenate([b, np.array([-np.inf] * D)])
    return b, cov_c, R
#    if mu.ndim == 1:
#        l_c = b - R.dot(mu)
#    elif mu.ndim == 2:
#        l_c = b - np.einsum('sd,nd->ns',R, mu)
#
#    cov_c = R.dot(cov.dot(R.T))
#    
#    return l_c, cov_c, R


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
    x_ = x[None, :] if x.ndim == 1 else x

    inv_cov_x = np.linalg.inv(cov_x) if x_.shape[1] > 1 else 1/cov_x
    inv_cov_z = np.linalg.inv(cov_z) if x_.shape[1] > 1 else 1/cov_z

    if A.ndim == 3:
        inv_cov_w = inv_cov_z + np.einsum('nds,dk,nkz->nsz',A, inv_cov_x, A)
    else:
        inv_cov_w = inv_cov_z + A.T.dot(inv_cov_x.dot(A))

    cov_w = np.linalg.inv(inv_cov_w) if inv_cov_w.ndim > 1 else 1/inv_cov_w
    
    if A.ndim == 3:
        mu_w = np.einsum('nsk,Nnk->Nns', cov_w, np.einsum('nds,dk,Nnk->Nns',
                                        A, inv_cov_x, x_[:, None, :] - b))
    else:
        mu_w = np.einsum('sk,Nk->Ns', cov_w, np.einsum('ds,dk,Nk->Ns',
                                    A, inv_cov_x, x_[:, None, :] - b))

    mu_w = mu_w[0] if x.ndim == 1 else mu_w
    
    return mu_w, cov_w


####################################################
#
#
#                   PHIS
#
#####################################################

def phis_w(ineq_w, mu, cov_w):

    ineqs = ineq_w + 0.
    ineqs[:, 0] += ineqs[:, 1:].dot(mu)
    phi0 = 0.
    phi1 = 0.
    phi2 = 0.

    ready = ineqs.shape[0] <= ineqs.shape[1]-1

    if ready:
        simplices = [range(len(ineqs))]
    else:
        v = np.array(get_vertices(ineqs))[:, 1:]
        simplices = get_simplices(v)
    for simplex in simplices:

        cones = [(ineqs, 1)] if ready else zip(*simplex_to_cones(v[simplex]))

        for ineqs_c, s in cones:

            l_c, cov_c, R_c = cones_to_rectangle(ineqs_c, cov_w)
        
            f, G = get_F_G(l_c, cov_c)
    
            phi0 += s * mvstdnormcdf(l_c, cov_c)
            phi1 += s * R_c.T.dot(f)
            M = (np.nan_to_num(l_c) * f - (cov_c * G).sum(1)) / np.diag(cov_c)
            phi2 += s * np.einsum('dk,db,bv->kv', R_c, G + np.diag(M), R_c)

    phi1 = cov_w.dot(phi1)
    phi2 = cov_w * phi0 + np.einsum('dk,dv,vl->kl',cov_w, phi2, cov_w)

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
    if b.ndim == 1:
        cov = cov_x + np.einsum('ds,sp,kp->dk',A, cov_z, A)
        return multivariate_normal.logpdf(x, mean=b, cov=cov)

    cov = cov_x + np.einsum('nds,sp,nkp->ndk',A, cov_z, A)
    kappas = np.array([multivariate_normal.logpdf(x, mean=b[i], cov=cov[i])
                                                       for i in range(len(b))])
    if kappas.ndim == 2:
        return kappas.T
    return kappas


##################################
def posterior(z, regions, x, As, Bs, cov_z, cov_x, input2signs):
    mu, cov = mu_sigma(x, As, Bs, cov_z, cov_x)
    kappas = np.exp(log_kappa(x, cov_x, cov_z, As, Bs))
    phis = phis_all([regions[r]['ineq'] for r in regions], mu, cov)
    indices = find_region(z, regions, input2signs)
    output = np.zeros(len(indices))
    for k in np.unique(indices):
        w = indices == k
        output[w] = multivariate_normal.pdf(z[w], mean=mu[k], cov=cov[k])
    px = (kappas * phis[0]).sum()
    output *= kappas[indices] / px

    alphas = kappas / px

    m0_w = phis[0] * alphas
    m1_w = phis[1] * alphas[:, None]
    m2_w = phis[2] * alphas[:, None, None] + np.einsum('nd,nk,n->ndk', mu, mu, m0_w)\
            + np.einsum('nd,nk->ndk', mu, m1_w) + np.einsum('nd,nk->ndk', mu, m1_w).transpose((0, 2, 1))
    m1_w += np.einsum('nd,n->nd', mu, m0_w)


    return output, m0_w, m1_w, m2_w




############################## ALGO 2
def marginal_moments(x, regions, cov_x, cov_z):

    As = np.array([regions[s]['Ab'][0] for s in regions])
    Bs = np.array([regions[s]['Ab'][1] for s in regions])

    mus, covs = mu_sigma(x, As, Bs, cov_z, cov_x) #(R D) (D D)

    log_kappas = log_kappa(x, cov_x, cov_z, As, Bs) #(R)
    
    ineqs = np.array([regions[r]['ineq'] for r in regions])
    
    phis = phis_all(ineqs, mus, covs)
    
    px = lse(log_kappas + np.log(np.maximum(phis[0], 1e-34)))

    renorm = (np.exp(log_kappas-log_kappas.max()) * phis[0]).sum()
    alphas = np.exp(log_kappas - log_kappas.max()) / renorm

    m0_w = softmax(np.log(np.maximum(phis[0], 1e-34)) + log_kappas)
    m1_w = phis[1] * alphas[:, None]
    m2_w = phis[2] * alphas[:, None, None]\
            + np.einsum('nd,nk,n->ndk', mus, mus, m0_w)\
            + np.einsum('nd,nk->ndk', mus, m1_w)\
            + np.einsum('nd,nk->nkd', mus, m1_w)
    m1_w += np.einsum('nd,n->nd', mus, m0_w)
    return px, m0_w, m1_w, m2_w


############################# evidence

