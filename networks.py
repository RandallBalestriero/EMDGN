import sys
sys.path.insert(0, "../SymJAX")
import numpy as np
import symjax as sj
from symjax import layers
import symjax.tensor as T
import utils


cov_b = 200
cov_W = 200

def init_weights(Ds, seed):
    np.random.seed(seed)
    Ws = [sj.initializers.glorot((j, i)) for j, i in zip(Ds[1:], Ds[:-1])]
    bs = [sj.initializers.normal((j,))/3 for j in Ds[1:]]
    return Ws, bs


def relu_mask(x, leakiness):
    if type(x) == list:
        return [relu_mask(xx, leakiness) for xx in x]
    return T.where(x > 0, 1., leakiness)



def encoder(X, latent_dim):
    layer = [layers.Dense(X, 32)]
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 32))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], latent_dim * 2))
    return layer


def compute_W_derivative(lhs, rhs, Ws, Qs, l):

    L = len(Qs)

    backward_As = [rhs]
    for i in range(0, l):
        backward_As.append(T.einsum('na,as,ns->na', Qs[i], Ws[i],
                                    backward_As[-1]))

    forward_As = [lhs]
    for i in range(l, L):
        forward_As.append(T.einsum('ns,sb,nb->nb', forward_As[-1],
                                   Ws[- i], Qs[-1]))

    return T.einsum('na,nb->nab', forward_As, backward_As)


def sum_no_i(l, i):
    no_i = 1 - T.one_hot(i, len(l))
    return l


def get_forward(Ws, Qs):
    """this function gives the slope matrix that forwards any pre activation
    of layer l to the output layer which is ::
        W^{L}Q^{L}_{\omega}W^{\ell}Q^{\ell}
    for the \ell element of the returned list. For the first one, is returns
    the entire A_{\omega} and for the last one it is the identity matrix
    """
    N = Qs[-1].shape[0]
    L = len(Qs)

    forward = [T.identity(Ws[-1].shape[0]) * T.ones((N, 1, 1))]
    for i in range(L):
        forward.append(T.einsum('ndb,bs,ns->nds', forward[-1], Ws[- 1 - i],
                                Qs[- 1 - i]))

    return forward[::-1]

def get_Abs(Ws, vs, Qs):
    
    """produces the pre activation feature maps of layer ell"""

    N = Qs[-1].shape[0]
    As = [Ws[0] * T.ones((N, 1, 1))]
    bs = [vs[0] * T.ones((N, 1))]

    for i in range(len(Qs)):
        As.append(T.einsum('db,nb,nbs->nds', Ws[i + 1], Qs[i], As[-1]))
        bs.append(T.einsum('db,nb,nb->nd', Ws[i + 1], Qs[i], bs[-1])\
                  + vs[i + 1])

    return As, bs






def create_vae(batch_size, Ds, seed, leakiness=0.1, lr=0.0002):

    x = T.Placeholder([batch_size, Ds[-1]], 'float32')

    # ENCODER
    enc = encoder(x, Ds[0])
    mu = enc[-1][:, :Ds[0]]
    logvar = enc[-1][:, Ds[0]:]
    var = T.exp(logvar)
 
    z = mu + T.exp(0.5 * logvar) * T.random.randn((batch_size, Ds[0]))
    z_ph = T.Placeholder((batch_size, Ds[0]), 'float32')

    # DECODER
    Ws, bs = init_weights(Ds, seed)

    Ws = [T.Variable(w) for w in Ws]
    bs = [T.Variable(b) for b in bs]
    logvar_x = T.Variable(T.zeros(1), name='logvar_x') 
    var_x = T.exp(logvar_x)

    h, h_ph = [z], [z_ph]
    for w, b in zip(Ws[:-1], bs[:-1]):
        h.append(T.matmul(h[-1], w.transpose()) + b)
        h.append(h[-1] * relu_mask(h[-1], leakiness))
        h_ph.append(T.matmul(h_ph[-1], w.transpose()) + b)
        h_ph.append(h_ph[-1] * relu_mask(h_ph[-1], leakiness))

    h.append(T.matmul(h[-1], Ws[-1].transpose()) + bs[-1])
    h_ph.append(T.matmul(h_ph[-1], Ws[-1].transpose()) + bs[-1])

    kl = 0.5 * (1 + logvar - var - mu ** 2).sum(1)
    # so average over L only for the below
    px = - 0.5 * (logvar_x + ((x - h[-1])**2 / var_x)).sum(1)
    loss = - (px + kl).mean()

    variables = Ws + bs + sj.layers.get_variables(enc) + [logvar_x]
    opti = sj.optimizers.Adam(loss, variables, lr)

    train = sj.function(x, outputs=loss, updates=opti.updates)
    g = sj.function(z_ph, outputs=h_ph[-1])
    params = sj.function(outputs = Ws + bs + [logvar_x])
    get_varx = sj.function(outputs = var_x)


    output = {'train': train, 'g':g, 'params':params}
    output['model'] = 'VAE'
    output['varx'] = get_varx

    def sample(n):
        samples = []
        for i in range(n // batch_size):
            samples.append(g(np.random.randn(batch_size, Ds[0])))
        return np.concatenate(samples)
    output['sample'] = sample
    return output



def create_fns(batch_size, R, Ds, seed, var_x, leakiness=0.1, lr=0.0002, var_z = None):

    x = T.Placeholder((Ds[0],), 'float32')
    X = T.Placeholder((batch_size, Ds[-1]), 'float32')

    q = T.Placeholder((np.sum(Ds[1:-1]),), 'float32')
    Q = T.Placeholder((R, np.sum(Ds[1:-1])), 'float32')
    
    m0 = T.Placeholder((batch_size, R), 'float32')
    m1 = T.Placeholder((batch_size, R, Ds[0]), 'float32')
    m2 = T.Placeholder((batch_size, R, Ds[0], Ds[0]), 'float32')

    cumulative_units = np.cumsum([0] + Ds[1:-1])

    Ws, vs = init_weights(Ds, seed)
    Ws = [T.Variable(w, name='W' + str(l)) for l, w in enumerate(Ws)]
    vs = [T.Variable(v, name='v' + str(l)) for l, v in enumerate(vs)]

    var_x = T.Variable(var_x)
    var_z = T.Variable(T.ones(Ds[0])) if var_z is None else var_z

    # create the placeholders
    Ws_ph = [T.Placeholder(w.shape, w.dtype) for w in Ws]
    vs_ph = [T.Placeholder(v.shape, v.dtype) for v in vs]
    var_x_ph = T.Placeholder(var_x.shape, var_x.dtype)

    # Compute the output of g(x)
    maps = [x]
    xqs = []
    masks = []
    
    for w, v in zip(Ws[:-1], vs[:-1]):
        
        pre_activation = T.matmul(w, maps[-1]) + v
        xqs.append(T.sign(pre_activation).reshape((1, -1)))
        masks.append(relu_mask(pre_activation, leakiness))
        maps.append(pre_activation * masks[-1])

    signs = T.concatenate(xqs, 1)
    maps.append(T.matmul(Ws[-1], maps[-1]) + vs[-1])

    qs = [q[cumulative_units[i]:cumulative_units[i + 1]].reshape((1, -1))
                    for i in range(len(cumulative_units) - 1)]
    Qs = [Q[:, cumulative_units[i]:cumulative_units[i + 1]]
                    for i in range(len(cumulative_units) - 1)]

    Axs, bxs = get_Abs(Ws, vs, relu_mask(xqs, leakiness))
    Aqs, bqs = get_Abs(Ws, vs, relu_mask(qs, leakiness))
    AQs, bQs = get_Abs(Ws, vs, relu_mask(Qs, leakiness))

    all_bxs = T.concatenate(bxs[:-1], 1)
    all_Axs = T.concatenate(Axs[:-1], 1)

    all_bqs = T.concatenate(bqs[:-1], 1)
    all_Aqs = T.concatenate(Aqs[:-1], 1)

    x_inequalities = T.hstack([all_bxs.transpose(), all_Axs[0]])\
                                                * signs.transpose()
    q_inequalities = T.hstack([all_bqs.transpose(), all_Aqs[0]]) * q[:, None]

    Bm0 = T.einsum('nd,Nn->Nd', bQs[-1], m0)
    Am1 = T.einsum('nds,Nns->Nd', AQs[-1], m1)
    xAm1Bm0 = X * (Am1 + Bm0)

    ABm1 = T.einsum('nds,nd,Nns->Nd', AQs[-1], bQs[-1], m1)
    Am2AT = T.diagonal(T.einsum('nds,Nnsc,npc->Ndp', AQs[-1], m2, AQs[-1]),
                        axis1=1, axis2=2)
    B2m0 = T.einsum('Nn,nd->Nd', m0, bQs[-1] ** 2)
    
    prior = sum([(v**2).sum() for v in vs], 0.) / cov_b\
            + sum([(w**2).sum() for w in Ws], 0.) / cov_W
    M2diag = T.diagonal(m2.sum(1), axis1=1, axis2=2)
    loss = - 0.5 * ((Ds[0] + Ds[-1]) * T.log(2 * np.pi) + T.log(var_x).sum()\
                                                 + T.log(var_z).sum())\
           - 0.5 * T.sum((X ** 2 - 2 * xAm1Bm0 + B2m0 + Am2AT + 2 * ABm1) / var_x, 1)\
            - 0.5 * T.sum(M2diag / var_z, 1) - 0.5 * prior
    mean_loss = - loss.mean()
    adam = sj.optimizers.NesterovMomentum(mean_loss, Ws, lr, 0.1)

    # update of var_x
    update_varx = (X ** 2 - 2 * xAm1Bm0 + B2m0 + Am2AT + 2 * ABm1).mean()\
                    * T.ones(Ds[-1])
    update_varz = M2diag.mean() * T.ones(Ds[0])
#    ppvar_x = update_varx / update_varx.min()
    # update for biases
    FQ = get_forward(Ws, relu_mask(Qs, leakiness))
    update_vs = []
    for i in range(len(vs)):
        
        # find which one to take
        to_take = update_vs[:i] + [T.zeros_like(vs[i])] + vs[i + 1:]
        
        # now we forward each bias to the x-space
        separated_bs = T.stack([T.einsum('nds,s->nd', FQ[u], to_take[u])
                                                    for u in range(len(FQ))])
        # sum the biases and apply the m0 scaling
        b_contrib = T.einsum('Lnd,Nn->nNd', separated_bs, m0)

        # get the linear contribution
        A_contrib = T.einsum('nds,Nns->nNd', AQs[-1], m1)

        # compute the residual and apply sigma
        error = (T.einsum('Nd,Nn->nNd', X, m0) - A_contrib - b_contrib)
        Error = T.einsum('nds,nNd->s', FQ[i], error / var_x)

        # compute the whitening matrix
        whiten = T.einsum('ndc,d,nds,Nn->cs', FQ[i], 1/var_x, FQ[i], m0) + batch_size * T.eye(Ds[i + 1]) / cov_b

        update_vs.append(T.matmul(T.linalg.inv(whiten), Error))
 
    # update for slopes
    update_Ws = []
    for i in range(len(Ws)):
        
        U = T.einsum('nds,d,ndc->nsc', FQ[i], 1/var_x, FQ[i])

        if i == 0:
            V = m2.mean(0)
        else:
            nAQs, nbQs = get_Abs(update_Ws[:i] + Ws[i:], vs, relu_mask(Qs, leakiness))
            V1 = T.einsum('nd,nq,Nn->Nndq', nbQs[i-1], nbQs[i-1], m0)
            V2 = T.einsum('nds,nqc,Nnsc->Nndq', nAQs[i-1], nAQs[i-1], m2)
            V3 = T.einsum('nds,nq,Nns->Nndq', nAQs[i-1], nbQs[i-1], m1)
            V4 = V3.transpose((0, 1, 3, 2))
            mask = relu_mask(Qs[i-1], leakiness)
            V = T.einsum('nd,Nndq,nq->ndq', mask, V1 + V2 + V3 + V4, mask)

        whiten = T.stack([T.kron(U[n], V[n].transpose())
                          for n in range(V.shape[0])])
        print(U, V, whiten)
        whiten_inv = T.linalg.inv(whiten.sum(0) + batch_size * T.eye(Ds[i] * Ds[i + 1]) / cov_W)

        # now we forward each bias to the x-space
        separated_bs = T.stack([T.einsum('nds,s->nd', FQ[u], vs[u])
                                            for u in range(i, len(FQ))])

        # sum the biases and apply the m0 scaling
        residual = (X[:, None, :] - separated_bs.sum(0)) / var_x

        if i == 0:
            rhs = m1
        else:
            rhs1 = T.einsum('nds,Nns->Nnd', nAQs[i-1], m1)
            rhs2 = T.einsum('nd,Nn->Nnd', nbQs[i-1], m0)
            rhs = T.einsum('nd,Nnd->Nnd', mask, rhs1 + rhs2)

        vector = T.einsum('ndc,Nnd,Nns->cs', FQ[i], residual, rhs)

        W = T.matmul(whiten_inv, vector.flatten()).reshape((Ds[i + 1], Ds[i]))

        update_Ws.append(W)

    updates = {**adam.updates}#dict(list(zip(vs, update_vs)))
    output = {'train':sj.function(Q, X, m0, m1, m2, outputs=mean_loss,
                                  updates=updates),
              'update_sigma':sj.function(Q, X, m0, m1, m2, outputs=mean_loss,
                                        updates = {var_x: update_varx}),
#                                                var_z: update_varz}),
              'update_vs':sj.function(Q, X, m0, m1, m2, outputs=mean_loss,
                                      updates = dict(list(zip(vs, update_vs)))),
              'loss':sj.function(Q, X, m0, m1, m2, outputs=mean_loss),
              'update_Ws':sj.function(Q, X, m0, m1, m2, outputs=mean_loss,
                                      updates = dict(list(zip(Ws, update_Ws)))),
              'signs2Ab': sj.function(q, outputs=[Aqs[-1][0], bqs[-1][0]]),
              'signs2ineq': sj.function(q, outputs=q_inequalities),
              'g': sj.function(x, outputs=maps[-1]),
              'input2all': sj.function(x, outputs=[maps[-1], Axs[-1][0],
                                       bxs[-1][0], x_inequalities, signs[0]]),
              'get_nll': sj.function(Q, X, m0, m1, m2, outputs=mean_loss),
              'assign': sj.function(*Ws_ph, *vs_ph, var_x_ph,
                                    updates=dict(zip(Ws + vs + [var_x],
                                             Ws_ph + vs_ph + [var_x_ph]))),
              'varx': sj.function(outputs=var_x),
              'varz': sj.function(outputs=var_z),
              'input2signs': sj.function(x, outputs=signs[0]),
              'S' : Ds[0], 'D':  Ds[-1], 'R': R, 'model': 'EM'}

    def sample(n):
        samples = []
        for i in range(n):
            samples.append(output['g'](np.random.randn(Ds[0])))
        return np.array(samples)
    
    output['sample'] = sample

    return output


def EM(model, DATA, epochs, n_iter):

    if model['model'] == 'VAE':
        L = []
        for e in range(epochs):
            for i in range(n_iter):
                L.append(model['train'](DATA))
        return L

    S, D, R = model['S'], model['D'], model['R']
    z = np.random.randn(S)/10

    m0 = np.zeros((DATA.shape[0], R))
    m1 = np.zeros((DATA.shape[0], R, S))
    m2 = np.zeros((DATA.shape[0], R, S, S))
    m_loss = []
    for e in range(epochs):
        output, A, b, inequalities, signs = model['input2all'](z)
        regions = utils.search_region(model['signs2ineq'], model['signs2Ab'],
                                      signs)
        batch_signs = np.pad(np.array(list(regions.keys())),
                             [[0, R - len(regions)], [0, 0]])

        print('regions', len(regions))
    
        varx = np.eye(D) * model['varx']()
        varz = np.eye(S) * model['varz']()
        print('varx', np.diag(varx))
        print('varz', np.diag(varz))
   
        m0 *= 0
        m1 *= 0
        m2 *= 0
        for i, x in enumerate(DATA):
            m0[i, :len(regions)], m1[i, :len(regions)], m2[i, :len(regions)] = utils.marginal_moments(x, regions, varx, varz)[1:]
    
        print('after E step', model['loss'](batch_signs, DATA, m0, m1, m2))
        for i in range(n_iter):
            m_loss.append(model['train'](batch_signs, DATA, m0, m1, m2))
            m_loss.append(model['update_vs'](batch_signs, DATA, m0, m1, m2))
#            m_loss.append(model['update_sigma'](batch_signs, DATA, m0, m1, m2))
        print('end M step', m_loss[-1])


#    m_loss.append(model['update_sigma'](batch_signs, DATA, m0, m1, m2))
    return m_loss



