from scipy.stats import kde, multivariate_normal
import cdd
import numpy as np
import itertools
from scipy.spatial import ConvexHull, Delaunay
from numpy.linalg import lstsq
from tqdm import tqdm


VERBOSE = 0

def get_simplices(vertices):  
    """compute the simplices from a convex polytope given in its
    V-representation

    vertices: array of shape (V, D) with V the number of vertices
    """
    return Delaunay(vertices).simplices

def one_step(M):
    A = np.vstack((M, np.random.rand(M.shape[1])))
    b = np.zeros(A.shape[0])
    b[-1] = 1
    vec = lstsq(A, b, rcond=None)[0]
    return np.vstack((M, vec / np.linalg.norm(vec, 2)))

def create_H(M):
    K = M.shape[0]
    for i in range(M.shape[1]-M.shape[0]):
        M = one_step(M)
    return M[K:]


def flip(A, i):
    sign = 1 - 2 * (np.arange(len(A)) == i).astype('float32')
    if A.ndim == 2:
        sign = sign[:, None]
    return A * sign

def search_region(A, b, S, signs):
    print(len(S), b.shape, A.shape)

    I = cdd.Matrix(np.hstack((b.reshape((-1, 1)), A)))
    I.rep_type = cdd.RepType.INEQUALITY 
    I = [i for i in range(len(A)) if i not in list(I.canonicalize()[1])]

    newregion = [I, list(signs[I])]

    if newregion in S:
        return
    
    S.append(newregion)

    for i in I:
        search_region(flip(A, i), flip(b, i), S, flip(signs, i))



def get_vertices(inequalitites):
    # create the matrix the inequalities are a matrix of the form
    # [b, -A] from b-Ax>=0
    m = cdd.Matrix(inequalities)
    m.rep_type = cdd.RepType.INEQUALITY
    return m.get_generators()


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
    m = cdd.Matrix(np.hstack([np.ones((len(vertices), 1)), vertices]))
    m.rep_type = cdd.RepType.GENERATOR
    v = np.array(cdd.Polyhedron(m).get_inequalities())
    A, b = v[:, 1:], -v[:, 0]
    K = A.shape[0]

    subsets = set()
    for n in range(1, K):
        subsets = subsets.union(set(itertools.combinations(set(range(K)), n)))
    F = len(vertices)
    signs = [(-1)**(F + 1)] + [(-1)**(len(J)+F+1) for J in subsets]
    sets = [(None, None)] + [(A[list(indices)], b[list(indices)])
            for indices in subsets]
    return sets, signs



def compute_mean_cov(x, A_w, b_w, mu_z, Sigma_z, Sigma_x):

    if len(x) > 1:
        inv_Sigma_x = np.linalg.inv(Sigma_x)
        inv_Sigma_z = np.linalg.inv(Sigma_z)
    else:
        inv_Sigma_x = 1/Sigma_x
        inv_Sigma_z = 1/Sigma_z
 
    inv_sigma_w = inv_Sigma_z + A_w.T.dot(inv_Sigma_x.dot(A_w))
    mu_w = inv_sigma_w.dot(A_w.T.dot(inv_Sigma_x.dot(x - b_w))\
                                     + inv_Sigma_z.dot(mu_z))
    if len(x) > 1:
        sigma_w = np.linalg.inv(inv_sigma_w)
    else:
        sigma_w = 1/inv_sigma_w
    return mu_w, sigma_w

############################# alpha computations
def compute_alphas(ineq_A, ineq_B, signs, x, mu_w, cov_w):

    alpha0 = 0.
    alpha1 = 0.
    alpha2 = 0.

    if len(ineq_B) < len(x):
        lower, mu_, cov_, R = planes_to_rectangle(ineq_A, ineq_B, mu_w,
                                                  cov_w)
        invR = np.linalg.inv(R)

        Phi = E_0(lower, mu_, cov_)
        Phi1 = E_X(lower, cov_)
        Phi2 = E_XXT(lower, cov_)

        alpha0 =  Phi
        alpha1 =  invR.dot(Phi1) + mu_ * Phi
        alpha2 =  invR.dot(Phi2.dot(invR.T))\
                +  np.outer(mu_, Phi).dot(invR.T)\
                +  invR.dot(np.outer(Phi, mu_)) +  np.outer(mu_, mu_) * Phi
        return alpha0, alpha1, alpha2

    m = cdd.Matrix(np.hstack((ineq_B[:, None], ineq_A)))
    m.rep_type = cdd.RepType.INEQUALITY
    v = np.array(cdd.Polyhedron(m).get_generators())[:, 1:]

    for simplex in get_simplices(v):
        a, signs = simplex_to_cones(v[simplex])
        for (A, b), sign in zip(a, signs):

            lower, mu_c, cov_c, R = planes_to_rectangle(A, b, mu_w, cov_w)
            invR = np.linalg.inv(R)

            Phi = E_0(lower, mu_c, cov_c)
            Phi1 = E_X(lower, cov_c)
            Phi2 = E_XXT(lower, cov_c)

            alpha0 += sign * Phi
            alpha1 += sign * invR.dot(Phi1) + sign * mu_c * Phi
            alpha2 += (invR.dot(Phi2.dot(invR.T))\
                    + np.outer(mu_c, Phi1).dot(invR.T)\
                    + invR.dot(np.outer(Phi1, mu_c))\
                    + np.outer(mu_c, mu_c) * Phi) * sign

    return alpha0, alpha1, alpha2

############################# kappa computations
def compute_kappa(x, mu_w, cov_w, cov_x, b_w):
    value = 0.5 * (mu_w * np.linalg.inv(cov_w).dot(mu_w)).sum()\
            - 0.5 * ((x - b_w) * np.linalg.inv(cov_x).dot(x - b_w)).sum()
    return np.exp(min(value, 2))

############################## ALGO 2
def algo2(x, ineq_A, ineq_B, signs, regions, sigma_x, mu_z, sigma_z,
          get_Ab):

    kappas = []
    all_As = []
    all_Bs = []
    alpha0 = []
    alpha1 = []
    alpha2 = []
    for (indices, flips) in regions:
        
        signs_ = np.copy(signs)
        signs_[indices] *= flips

        A_w, b_w = get_Ab(signs_)
        all_As.append(A_w)
        all_Bs.append(b_w)

        mu_w, cov_w = compute_mean_cov(x, A_w, b_w, mu_z, sigma_z, sigma_x)
     
        alphas = compute_alphas(ineq_A[indices], ineq_B[indices],
                                signs[indices] * flips, x, mu_w, cov_w)
        alpha0.append(alphas[0])
        alpha1.append(alphas[1])
        alpha2.append(alphas[2])
    
        kappas.append(compute_kappa(x, mu_w, cov_w, sigma_x, b_w))

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
    px = np.exp(-0.5 * (mu_z * np.linalg.inv(sigma_z).dot(mu_z)).sum())\
            * ((2*np.pi) ** (len(mu_z) + len(x))\
            * np.linalg.det(sigma_z) * np.linalg.det(sigma_x))**-0.5\
            * (kappas * alpha0).sum()
    return px, m1, m2, m0_w, m1_w, m2_w



############################# evidence

