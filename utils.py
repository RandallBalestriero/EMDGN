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
import scipy
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

def reduce_ineq(A, b):
    if A.shape[1] == 1:
        A_ = A[:, 0]
        array = - b / A_
        pos = np.where(A_ > 0)[0]
        neg = np.where(A_ < 0)[0]
        lower = pos[np.argmax(array[pos])]
        upper = neg[np.argmin(array[neg])]
#        if len(pos) > 0:
#            lower = pos[np.argmax(array[pos])]
#        else:
#            lower = None
#        if len(neg) > 0:
#            upper = neg[np.argmin(array[neg])]
#        else:
#            upper = None
#
#        if lower is None:
#            return [upper]
#        if upper is None:
#            return [lower]
        return [lower, upper]
    M = cdd.Matrix(np.hstack([b[:, None], A]))
    M.rep_type = cdd.RepType.INEQUALITY
    feasibles = set(list(np.nonzero(np.linalg.norm(A,axis=1) > 1e-6)[0]))
    I = feasibles - set(M.canonicalize()[1])
    return list(I)


def search_region_sample(input2signs):
    z = np.random.randn(10000, 1) * 2
    signs = []
    for i in range(len(z)):
        signs.append(input2signs(z[i]))
    signs = set([tuple(a) for a in signs])
    return signs



def search_region(signs2ineq, signs2Ab, signs, input2signs=None):
    all_signs = set()
    # init the to_visit
    to_visit=set([tuple(signs)])
    # search all the regions (their signs)
    #print('searching regions')
    while True:
        if len(to_visit) == 0:
            break
        all_signs = all_signs.union(to_visit)
        #print('current all_signs', all_signs)
        neighbours = set()
        for s in to_visit:
            
            # find neighbour
            ineqs = bound_ineq(signs2ineq(np.array(s)))
            I = reduce_ineq(ineqs[:,:-1], ineqs[:,-1])
            I = [i for i in I if i<len(s)]
            #print('current regions', s, ':',get_vertices(ineqs[:,:-1], ineqs[:,-1]))
            #print('to visit', I)
            if len(I) == 0:
                continue
            F = np.ones((len(I), len(s)))
            F[np.arange(len(I)), I] = - 1
            neighbours = neighbours.union(set([tuple(r) for r in F * s]))
        to_visit = neighbours - all_signs

    # now set up S
    S = dict()
#    all_signs = search_region_sample(input2signs)
    for s in all_signs:
        ineq = bound_ineq(signs2ineq(s))
        v = get_vertices(ineq[:, :-1], ineq[:,-1])
        if v[0] > v[1] - 0.0001:
            continue
        S[s] = {'ineq': ineq, 'Ab': signs2Ab(s)}
    return S



def bound_ineq(ineq):
    if ineq.shape[1] == 2 :
        return np.vstack([ineq, np.array([[1, 18],[-1, 18]])])
    else:
        return np.vstack([ineq, np.array([[1, 0, 20],[0, 1, 20],
                                         [-1, 0, 20],[0, -1, 20]])])
 


def get_vertices(A,b):
    # input is of the form [A, b] s.t. Ax+b >= 0
    # create the matrix the inequalities are a matrix of the form
    # [b, -A] from Ax<=b
    if A.shape[1] == 1:
        A_ = A[:, 0]
        lower = np.max(-b[A_ > 0] / A_[A_ > 0])
        upper = np.min(-b[A_ < 0] / A_[ A_ < 0])
        if lower >= upper:
            return None
        return np.array([lower, upper])
    m = cdd.Matrix(np.hstack([b.reshape((-1,1)),  A]))
    m.rep_type = cdd.RepType.INEQUALITY
    try:
        return np.sort(np.array(cdd.Polyhedron(m).get_generators())[:, 1])
    except:
        return None


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

    A, b = ineqs[:, :-1], ineqs[:, -1]
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

def phis_w(ineq, mu, cov_w):

    A = ineq[:, :-1]
    B = ineq[:, -1]

    # instead of integrating a non centered gaussian on w 
    # we integrate a centered Gaussian on w-mu. This is equivalent to 
    # adding mu to the bias of the inequality system
    B_mu = B + A.dot(mu)

    if len(cov_w) > 1:
        norm_vector = np.linalg.norm(ineq[:, :-1], axis=1, keepdims=True)
        system = np.hstack([-A, -B_mu.reshape((-1, 1))])
        c = np.zeros((A.shape[1] + 1,))
        c[-1] = -1
        res = scipy.optimize.linprog(c, A_ub=np.hstack((-A, norm_vector)), b_ub=B_mu)
        inter = scipy.spatial.HalfspaceIntersection(system, res.x[:-1])
        vertices = inter.dual_vertices
        simplices = Delaunay(vertices).simplices
    else:#VERTICES SHOULD BE ORDERED
        vertices = get_vertices(A, B_mu)
        phi0 = scipy.stats.norm.sf(vertices[0], scale=np.sqrt(cov_w))\
                - scipy.stats.norm.sf(vertices[1], scale=np.sqrt(cov_w))
        f1 = scipy.stats.norm.pdf(vertices[0], scale=np.sqrt(cov_w))
        f2 = scipy.stats.norm.pdf(vertices[1], scale=np.sqrt(cov_w))#get_F_G(vertices[[1]], cov_w)[0]

        phi1 = cov_w * (f1 - f2)
        phi2 = cov_w * (phi0 + vertices[[0]] * f1 - vertices[[1]] * f2)
        ###
        phi2 = phi0 * mu ** 2 + 2 * mu * phi1 + phi2# * cov_w#**2
        phi1 += mu * phi0

        return phi0, phi1, phi2

    
    # we initialize the accumulators
    phi0, phi1, phi2 = 0., 0., 0.

    for simplex in simplices:

        if len(cov_w) == 1:
            lows = vertices[[0]], vertices[[1]]
            Rs = - np.ones((1, 1)), np.ones((1, 1))
            signs = 1, -1
        else:
            if ineqs.shape[0] < ineqs.shape[1]:
                low, Rs = cones_to_rectangle(ineqs, cov_w)
                lows = [low]
                Rs = [Rs]
                signs = [1]
            else: #TODO
                cones = zip(*simplex_to_cones(vertices[simplex].reshape((-1,1))))

        for l_c, R_c, s in zip(lows, Rs, signs):

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
    print(np.einsum('nds,sp,nkp->ndk',A, cov_z, A)[-1, -1,-2:])
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
    ineqs = [regions[r]['ineq'] for r in regions]
    P0, P1, P2 = [], [], []
    for mu in mus:
        p0, p1, p2 = phis_all(ineqs, mu, covs)
        P0.append(p0)
        P1.append(p1)
        P2.append(p2)

    phis = [np.array(P0), np.array(P1), np.array(P2)]
    phis[0] += 1e-42#14#= np.maximum(phis[0], 1e-15)

    # compute marginal
    px = lse(log_kappas + np.log(phis[0]), axis=1) # (N)

    # compute per region moments
    alphas = np.exp(log_kappas - log_kappas.max(1, keepdims=True))\
            / (np.exp(log_kappas - log_kappas.max(1, keepdims=True)) * phis[0]).sum(1, keepdims=True)
#    alphas = softmax(log_kappas + np.log(np.maximum(phis[0], 1e-48)), 1) / (np.maximum(phis[0], 1e-48))

    m0_w = softmax(log_kappas + np.log(phis[0]), 1)
    if 0:#len(cov_z) == 1:
        m2_w = m0_w[:, :, None, None] * mus[:, :, :, None] ** 2\
            + ((2 * mus * phis[1])[:, :, :, None]\
            + phis[2]) * alphas[:, :, None, None]
        m1_w = phis[1] * alphas[:, :, None] + mus * m0_w[:, :, None]
    else:
        m0_w = phis[0] * alphas
        m1_w = phis[1] * alphas[:, :, None]
        m2_w = phis[2] * alphas[:, :, None, None]

    return px, m0_w, m1_w, m2_w
