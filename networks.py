import sys
sys.path.insert(0, "../SymJAX")
import numpy as np
import symjax as sj
from symjax import layers
import symjax.tensor as T
import utils


def init_weights(Ds, seed):
    np.random.seed(seed)
    Ws = [sj.initializers.glorot((j, i)) for j, i in zip(Ds[1:], Ds[:-1])]
    bs = [sj.initializers.he((j,)) for j in Ds[1:]]
    return Ws, bs


def relu_mask(x, leakiness):
    return T.where(x > 0, 1., leakiness)

def encoder(X, latent_dim):
    layer = [layers.Dense(X, 32)]
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 32))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], latent_dim * 2))
    return layer


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

    output = {'train': train, 'g':g, 'x':x, 'z':z, 'params':params}
    output['model'] = 'VAE'
    def sample(n):
        samples = []
        for i in range(n // batch_size):
            samples.append(g(np.random.randn(batch_size, Ds[0])))
        return np.concatenate(samples)
    output['sample'] = sample
    return output



def create_fns(batch_size, R, Ds, seed, var_x, leakiness=0.1, lr=0.0002):

    input = T.Placeholder((Ds[0],), 'float32')
    in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')
    
    batch_in_signs = T.Placeholder((R, np.sum(Ds[1:-1])), 'bool')
    x = T.Placeholder((batch_size, Ds[-1]), 'float32')
    m0 = T.Placeholder((batch_size, R), 'float32')
    m1 = T.Placeholder((batch_size, R, Ds[0]), 'float32')
    m2 = T.Placeholder((batch_size, R, Ds[0], Ds[0]), 'float32')

    cumulative_units = np.concatenate([[0], np.cumsum(Ds[:-1])])

    Ws, bs = init_weights(Ds, seed)

    # create the variables
    Ws = [T.Variable(w) for w in Ws]
    bs = [T.Variable(b) for b in bs]
    var_x = T.Variable(var_x)
    var_z = T.Variable(T.ones(Ds[0]))

    # create the placeholders
    Ws_ph = [T.Placeholder(w.shape, w.dtype) for w in Ws]
    bs_ph = [T.Placeholder(b.shape, b.dtype) for b in bs]
    var_x_ph = T.Placeholder(var_x.shape, var_x.dtype)

    A_w = [T.eye(Ds[0])]
    B_w = [T.zeros(Ds[0])]
    
    A_q = [T.eye(Ds[0])]
    B_q = [T.zeros(Ds[0])]
    
    batch_A_q = [T.eye(Ds[0]) * T.ones((R, 1, 1))]
    batch_B_q = [T.zeros((R, Ds[0]))]
    
    maps = [input]
    signs = []
    masks = [T.ones(Ds[0])]
    in_masks = relu_mask(T.concatenate([T.ones(Ds[0]), in_signs]), leakiness)
    batch_in_masks = relu_mask(T.concatenate([T.ones((R, Ds[0])),
                                              batch_in_signs], 1), leakiness)
    for w, b in zip(Ws[:-1], bs[:-1]):
        
        pre_activation = T.matmul(w, maps[-1]) + b
        signs.append(T.sign(pre_activation))
        masks.append(relu_mask(pre_activation, leakiness))
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

    Am1 = T.einsum('qds,nqs->nd', batch_A_q, m1)
    ABm1 = T.einsum('qds,qd,nqs->nd', batch_A_q, batch_B_q, m1)
    Bm0 = T.einsum('qd,nq->nd', batch_B_q, m0)
    inner = 2 * (ABm1 - x * (Am1 + Bm0))

    Am2AT = T.einsum('qds,nqsu,qpu->ndp', batch_A_q, m2, batch_A_q)
    B2m0 = T.einsum('nq,qd->nd', m0, batch_B_q ** 2)
    squares = x ** 2 + B2m0 + T.diagonal(Am2AT, axis1=1, axis2=2)

    pz = T.diagonal(m2, axis1=2, axis2=3).sum(1)

    cst = (Ds[0] + Ds[-1]) * T.log(2 * np.pi) + T.log(var_x).sum() + T.log(var_z).sum()

    loss = 0.5 * (cst + T.sum((inner + squares) / var_x, 1)\
                      + T.sum(pz / var_z, 1))
    mean_loss = loss.mean()
    adam = sj.optimizers.NesterovMomentum(mean_loss, Ws + bs, lr, 0.5)
    update_varx =  (inner + squares).mean() * T.ones(Ds[-1])
    update_varz =  pz.mean(0) * T.ones(Ds[0])
    updates = {**adam.updates, var_x: update_varx}
    output = {'train':sj.function(batch_in_signs, x, m0, m1, m2,
                                  outputs=mean_loss, updates=updates),
              'signs2Ab': sj.function(in_signs, outputs=[A_q[-1], B_q[-1]]),
              'signs2q': sj.function(in_signs, outputs=inequalities_code),
              'g': sj.function(input, outputs=maps[-1]),
              'input2all': sj.function(input, outputs=[maps[-1], A_w[-1],
                                       B_w[-1], inequalities, signs]),
              'get_nll': sj.function(batch_in_signs, x, m0, m1, m2,
                                     outputs=mean_loss),
              'assign': sj.function(*Ws_ph, *bs_ph, var_x_ph,
                                    updates=dict(zip(Ws + bs + [var_x],
                                             Ws_ph + bs_ph + [var_x_ph]))),
              'varx': sj.function(outputs=var_x),
              'varz': sj.function(outputs=var_z),
              'input2signs': sj.function(input, outputs=signs),
              'S' : Ds[0], 'D':  Ds[-1], 'R': R, 'model': 'EM'}
    def sample(n):
        samples = []
        for i in range(n):
            samples.append(output['g'](np.random.randn(Ds[0])))
        return np.array(samples)
    output['sample'] = sample

    return output


def EM(model, DATA, n_iter):

    if model['model'] == 'VAE':
        L = []
        for i in range(n_iter):
            L.append(model['train'](DATA))
        return L

    S, D, R = model['S'], model['D'], model['R']
    z = np.random.randn(S)/10
    output, A, b, inequalities, signs = model['input2all'](z)
    
    regions = utils.search_region(model['signs2q'], model['signs2Ab'], signs)
    print('regions', len(regions))

    varx = np.eye(D) * model['varx']()
    varz = np.eye(S) * model['varz']()
    print('varx', np.diag(varx))
    print('varz', np.diag(varz))
    m0 = np.zeros((DATA.shape[0], len(regions)))
    m1 = np.zeros((DATA.shape[0], len(regions), S))
    m2 = np.zeros((DATA.shape[0], len(regions), S, S))

    for i, x in enumerate(DATA):
        m0[i], m1[i], m2[i] = utils.marginal_moments(x, regions, varx, varz)[1:]

    P = R - len(regions)
    assert P >= 0
    m0 = np.pad(m0, [[0, 0], [0, P]])
    m1 = np.pad(m1, [[0, 0], [0, P], [0, 0]])
    m2 = np.pad(m2, [[0, 0], [0, P], [0, 0], [0, 0]])
    batch_signs = np.pad(np.array(list(regions.keys())), [[0, P], [0, 0]])
    
    m_loss = []
    for i in range(n_iter):
        m_loss.append(model['train'](batch_signs, DATA, m0, m1, m2))
    return m_loss



