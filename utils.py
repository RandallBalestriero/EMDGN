from scipy.stats import kde, multivariate_normal
import cdd
import numpy as np
import itertools
from scipy.spatial import ConvexHull, Delaunay
from numpy.linalg import lstsq
from tqdm import tqdm
import symjax as sj
import symjax.tensor as T
VERBOSE = 0

def create_fns(input, in_signs, Ds):

    cumulative_units = np.concatenate([[0], np.cumsum(Ds[:-1])])
    
    Ws = [sj.initializers.he((j, i))/1 for j, i in zip(Ds[1:], Ds[:-1])]
    bs = [sj.initializers.he((j,))/3 for j in Ds[1:]]

    A_w = [T.eye(Ds[0])]
    B_w = [T.zeros(Ds[0])]
    
    A_q = [T.eye(Ds[0])]
    B_q = [T.zeros(Ds[0])]
    
    maps = [input]
    signs = []
    masks = [T.ones(Ds[0])]
    in_masks = T.where(T.concatenate([T.ones(Ds[0]), in_signs]) > 0, 1.,
                                     0.3)

    for w, b in zip(Ws[:-1], bs[:-1]):
        
        pre_activation = T.matmul(w, maps[-1]) + b
        signs.append(T.sign(pre_activation))
        masks.append(T.where(pre_activation > 0, 1., 0.3))

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



def lse(x):
    x_max = x.max()
    return np.log(np.sum(np.exp(x - x_max))) + x_max

def find_region(x, regions, f):
    x_signs = []
    for x_ in x:
        x_signs.append(f(x_)[-1])
    x_signs = np.array(x_signs)

    return np.equal(x_signs[:, None, :], np.array(list(regions.keys()))).prod(2).argmax(1)

def in_region(x, ineq_A, ineq_B):
    if ineq_A is None:
        return np.ones(x.shape[0]).astype('bool')
    return (np.dot(x, ineq_A.T) + ineq_B >= 0).prod(1)



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
    ineq = all_g(signs)

    M = cdd.Matrix(np.hstack([ineq[:, [0]], ineq[:, 1:]]))
    M.rep_type = cdd.RepType.INEQUALITY
    redundant = set(M.canonicalize()[1])
    I = list(set(range(len(signs))) - redundant)
    F = np.ones((len(I), len(signs)))
    F[np.arange(len(I)), I] = -1
    return F * signs, np.array(M)



def search_region(all_g, signs, max_depth=9999999999999):
    S = dict()
    parents=[signs]
    for d in range(max_depth+1):
        if len(parents) == 0:
            return S
        children = []
        for s in parents:
            neighbours, M = find_neighbours(all_g, s)
            S[tuple(s)] = M
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


def univariate_gaussian_integral(low, high, mu, cov):
    return multivariate_normal.cdf(high, mu=mu, cov=cov) - multivariate_normal.cdf(lower, mu=mu, cov=cov)

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


def E_1(lower, cov):
    return mvstdnormcdf(lower, cov)

def E_0(lower, mu, cov):
    return mvstdnormcdf(lower-mu, cov)


def E_X(lower, cov):
    """compute the first moment"""

    D = len(lower)
    f = np.zeros(D)
    valid = np.nonzero(np.isfinite(lower))[0]

    for k, low in zip(valid, lower[valid]):

        keeping = np.arange(D) != k

        cov_ = np.delete(np.delete(cov, k, 0), k, 1)
        covk = cov_ - np.outer(cov[keeping, k], cov[k, keeping]) / cov[k, k]

        muk = cov[keeping, k] * low / cov[k, k]

        f[k] = F(low, cov[k, k], lower[keeping], muk, covk)

    return np.matmul(cov, f)


def E_Y(lower, mu, cov):
    return E_X(lower - mu, cov) + mu * E_1(lower - mu, cov)


def E_XXT(lower, cov):
    """compute the moment 1 given a set of planes inequality
    smaller of equal to d
    """

    alpha = mvstdnormcdf(lower, cov)

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
    return (cov*first).dot(cov.T) + alpha * cov



def E_YYT(lower, mu, cov):
    a = E_X(lower - mu, cov)
    b = E_XXT(lower - mu, cov)+np.outer(a, mu) + np.outer(mu, a)\
            +np.outer(mu, mu) * E_1(lower - mu, cov)
    return b

def E_YmYmT(lower, upper, mu, cov):
    return E_XXT(lower-mu, cov)


def planes_to_rectangle(A, b, mu, cov):
    # first the general case without constraints
    if A is None:
        lower = np.array([-np.inf] * len(mu))
        return lower, mu, cov, np.eye(len(lower))

    b /= np.linalg.norm(A, 2, 1)
    A /= np.linalg.norm(A, 2, 1, keepdims=True)

    if A.shape[0] == A.shape[1]:
        R = A
    else:
        R = np.vstack([A, create_H(A).dot(np.linalg.inv(cov))])

    newmu = R.dot(mu)
    newcov = R.dot(cov.dot(R.T))

    lower = np.concatenate([b, np.array([-np.inf] * (len(mu) - len(b)))])
    
    return lower, newmu, newcov, R


def simplex_to_cones(vertices):
    S = len(vertices[0])
    m = cdd.Matrix(np.hstack([np.ones((len(vertices), 1)), vertices]))
    m.rep_type = cdd.RepType.GENERATOR
    v = np.array(cdd.Polyhedron(m).get_inequalities())
    A, b = v[:, 1:], v[:, 0]
    K = A.shape[0]

    subsets = set()
    for n in range(1, K):
        subsets = subsets.union(set(itertools.combinations(set(range(K)), n)))
    F = len(vertices)
    signs = [(-1)**S] + [(-1)**(len(J) + S) for J in subsets]
    sets = [(None, None)] + [(A[list(indices)], b[list(indices)])
            for indices in subsets]
    return sets, signs



def compute_mean_cov(x, A_w, b_w, mu_z, sigma_z, sigma_x):

    if len(x) > 1:
        inv_sigma_x = np.linalg.inv(sigma_x)
        inv_sigma_z = np.linalg.inv(sigma_z)
    else:
        inv_sigma_x = 1/sigma_x
        inv_sigma_z = 1/sigma_z
 
    inv_sigma_w = inv_sigma_z + A_w.T.dot(inv_sigma_x.dot(A_w))
    mu_w = inv_sigma_w.dot(A_w.T.dot(inv_sigma_x.dot(x - b_w))\
                                     + inv_sigma_z.dot(mu_z))
    if len(x) > 1:
        sigma_w = np.linalg.inv(inv_sigma_w)
    else:
        sigma_w = 1/inv_sigma_w
    return mu_w, sigma_w

############################# alpha computations
def compute_alphas(ineqs, x, mu_w, sigma_w):

    if ineqs.shape[0] <= len(x):
        lower, mu_, cov_, R = planes_to_rectangle(ineqs[:, 1:], ineqs[:, 0],
                                                   mu_w, sigma_w)
        invR = np.linalg.inv(R)

        const = (2 * np.pi)**(len(x)/2) * np.sqrt(np.abs(np.linalg.det(cov_)))
        Phi = E_0(lower, mu_, cov_) * const
        Phi1 = E_X(lower, cov_) * const
        Phi2 = E_XXT(lower, cov_) * const

        alpha1 =  invR.dot(Phi1) + mu_ * Phi
        alpha2 =  invR.dot(Phi2.dot(invR.T))\
                +  np.outer(mu_, Phi1).dot(invR.T)\
                +  invR.dot(np.outer(Phi1, mu_)) +  np.outer(mu_, mu_) * Phi
        return Phi, alpha1, alpha2

    alpha0 = 0.
    alpha1 = 0.
    alpha2 = 0.


    v = np.array(get_vertices(ineqs))[:, 1:]

    for simplex in get_simplices(v):
        a, signs = simplex_to_cones(v[simplex])
        for (A, b), sign in zip(a, signs):

            lower, mu_c, cov_c, R = planes_to_rectangle(A, b, mu_w, sigma_w)
            invR = np.linalg.inv(R)
            const = (2 * np.pi)**(len(x)/2) * np.sqrt(np.abs(np.linalg.det(cov_c)))

            Phi = E_0(lower, mu_c, cov_c) * const
            print('PHI', sign, E_0(lower, mu_c, cov_c), const)
            Phi1 = E_X(lower, cov_c) * const
            Phi2 = E_XXT(lower, cov_c) * const
            alpha0 += sign * Phi
            alpha1 += sign * (invR.dot(Phi1) + mu_c * Phi)
            alpha2 += (invR.dot(Phi2.dot(invR.T))\
                    + np.outer(mu_c, Phi1).dot(invR.T)\
                    + invR.dot(np.outer(Phi1, mu_c))\
                    + np.outer(mu_c, mu_c) * Phi) * sign
    print('TOTAL', alpha0)
    return alpha0, alpha1, alpha2

############################# kappa computations
def compute_log_kappa(x, mu_w, sigma_w, sigma_x, b_w):
    value = mu_w.dot(np.linalg.inv(sigma_w).dot(mu_w))
    value -=(x - b_w).dot(np.linalg.inv(sigma_x).dot(x - b_w))
#    print('------ - - - - KAPPA')
#    print('x-b_w', x - b_w, '\nmu_w', mu_w, '\ninv_sigma_w', 
#            np.linalg.inv(sigma_w))
#    print('value=', value)
    return 0.5 * value

############################# get all A,B
def get_AB(ineq_A, ineq_B, signs, regions, get_Ab):

    all_As = []
    all_Bs = []
    for (indices, flips) in regions:
        
        signs_ = np.copy(signs)
        signs_[indices] *= flips

        A_w, b_w = get_Ab(signs_)
        all_As.append(A_w)
        all_Bs.append(b_w)

    return all_As, all_Bs






############################## ALGO 2
def algo2(x, regions, sigma_x, mu_z, sigma_z,
          get_Ab, get_ineq):

    kappas = []
#    all_As = []
#    all_Bs = []
    alpha0 = []
    alpha1 = []
    alpha2 = []

    for flips in regions.keys():
        
        A_w, b_w = get_Ab(np.array(flips))

#        all_As.append(A_w)
#        all_Bs.append(b_w)

        mu_w, sigma_w = compute_mean_cov(x=x, A_w=A_w, b_w=b_w, mu_z=mu_z,
                                       sigma_z=sigma_z, sigma_x=sigma_x)
     
        alphas = compute_alphas(regions[flips], x=x, mu_w=mu_w, sigma_w=sigma_w)

        alpha0.append(alphas[0])
        alpha1.append(alphas[1])
        alpha2.append(alphas[2])
    
        kappas.append(compute_log_kappa(x, mu_w=mu_w, sigma_w=sigma_w,
                                        sigma_x=sigma_x, b_w=b_w))

    kappas = np.nan_to_num(np.array(kappas))
    alpha0 = np.array(alpha0)
    alpha1 = np.array(alpha1)
    alpha2 = np.array(alpha2)
    renorm = kappas / (kappas * alpha0).sum()
    m0_w = alpha0 * renorm
    m1_w = alpha1 * renorm[:, None]
    m2_w = alpha2 * renorm[:, None, None]
    m1 = m1_w.sum(0)
    m2 = m2_w.sum(0)
    print(alpha0)
    px = -0.5 * (mu_z * np.linalg.inv(sigma_z).dot(mu_z)).sum()\
            - 0.5 * np.log((2*np.pi) ** (len(mu_z) + len(x))\
            * np.abs(np.linalg.det(sigma_z) * np.linalg.det(sigma_x)))
    try:
        px += lse(kappas[alpha0 > 1e-7] + np.log(alpha0[alpha0 > 1e-7]))
    except:
        px = 0
#    ps = 
    return px, m1, m2, m0_w, m1_w, m2_w



############################# evidence

