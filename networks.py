import sys
sys.path.insert(0, "../SymJAX")
import numpy as np
import symjax as sj
from symjax import layers
import symjax.tensor as T
import utils
from tqdm import tqdm

cov_b = 100000
cov_W = 100000

def init_weights(Ds, seed, scaler=1):
    np.random.seed(seed)
    Ws = [sj.initializers.he((i, j)) * scaler for i, j in zip(Ds[1:], Ds[:-1])]
    bs = [sj.initializers.normal((j,))/10 * scaler for j in Ds[1:]]
    return Ws, bs


def relu_mask(x, leakiness):
    if type(x) == list:
        return [relu_mask(xx, leakiness) for xx in x]
    return T.where(x >= 0, 1., leakiness)



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

    n = Qs[-1].shape[0]
    As = [Ws[0] * T.ones((n, 1, 1))]
    bs = [vs[0] * T.ones((n, 1))]
    for i in range(len(Qs)):
        As.append(T.einsum('db,nb,nbs->nds', Ws[i + 1], Qs[i], As[-1]))
        bs.append(T.einsum('db,nb,nb->nd', Ws[i + 1], Qs[i], bs[-1])\
                  + vs[i + 1])

    return As, bs



def create_glo(batch_size, Ds, seed, leakiness=0.1, lr=0.0002, scaler=1,
               GLO=False):

    x = T.Placeholder([batch_size, Ds[-1]], 'float32')
    z = T.Variable(T.random.randn((batch_size, Ds[0])))
    logvar_x = T.Variable(T.ones(1))

    # DECODER
    Ws, bs = init_weights(Ds, seed, scaler)
    Ws = [T.Variable(w) for w in Ws]
    bs = [T.Variable(b) for b in bs]
    h = [z]
    for w, b in zip(Ws[:-1], bs[:-1]):
        h.append(T.matmul(h[-1], w.transpose()) + b)
        h.append(h[-1] * relu_mask(h[-1], leakiness))
    h.append(T.matmul(h[-1], Ws[-1].transpose()) + bs[-1])

    # LOSS
    prior = sum([T.sum(w**2) for w in Ws], 0.) / cov_W + sum([T.sum(v**2) for v in bs[:-1]], 0.) / cov_b
    if GLO:
        loss = T.sum((x - h[-1])**2) / batch_size + prior
        variables = Ws + bs
    else:
        loss = Ds[-1] * logvar_x.sum() + T.sum((x - h[-1])**2 / T.exp(logvar_x)) / batch_size + (z**2).sum() / batch_size + prior
        variables = Ws + bs + [logvar_x]

    prior = sum([(b**2).sum() for b in bs], 0.) / cov_b\
            + sum([(w**2).sum() for w in Ws], 0.) / cov_W
 
    opti = sj.optimizers.Adam(loss + prior, lr, params=variables)
    infer = sj.optimizers.Adam(loss, lr, params=[z])

    estimate = sj.function(x, outputs=z, updates=infer.updates)
    train = sj.function(x, outputs=loss, updates=opti.updates)
    lossf = sj.function(x, outputs=loss)
    params = sj.function(outputs = Ws + bs + [T.ones(Ds[-1]) * T.exp(logvar_x)])

    output = {'train': train, 'estimate':estimate, 'params':params}
    output['reset'] = lambda v: z.assign(v)
    if GLO:
        output['model'] = 'GLO'
    else:
        output['model'] = 'HARD'
    output['loss'] = lossf
    output['kwargs'] = {'batch_size': batch_size, 'Ds':Ds, 'seed':seed,
                    'leakiness':leakiness, 'lr':lr, 'scaler':scaler}
    return output



def create_vae(batch_size, Ds, seed, leakiness=0.1, lr=0.0002, scaler=1):

    x = T.Placeholder([batch_size, Ds[-1]], 'float32')

    # ENCODER
    enc = encoder(x, Ds[0])
    mu = enc[-1][:, :Ds[0]]
    logvar = enc[-1][:, Ds[0]:]
    var = T.exp(logvar)
 
    z = mu + T.exp(0.5 * logvar) * T.random.randn((batch_size, Ds[0]))
    z_ph = T.Placeholder((batch_size, Ds[0]), 'float32')

    # DECODER
    Ws, bs = init_weights(Ds, seed, scaler)

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
    px = - 0.5 * (logvar_x + ((x - h[-1])**2 / var_x)).sum(1)
    loss = - (px + kl).mean()

    variables = Ws + bs + sj.layers.get_variables(enc) + [logvar_x]
    opti = sj.optimizers.Adam(loss, lr, params=variables)

    train = sj.function(x, outputs=loss, updates=opti.updates)
    g = sj.function(z_ph, outputs=h_ph[-1])
    params = sj.function(outputs = Ws + bs + [T.exp(logvar_x) * T.ones(Ds[-1])])
    get_varx = sj.function(outputs = var_x)


    output = {'train': train, 'g':g, 'params':params}
    output['model'] = 'VAE'
    output['varx'] = get_varx
    output['kwargs'] = {'batch_size': batch_size, 'Ds':Ds, 'seed':seed,
                    'leakiness':leakiness, 'lr':lr, 'scaler':scaler}
    def sample(n):
        samples = []
        for i in range(n // batch_size):
            samples.append(g(np.random.randn(batch_size, Ds[0])))
        return np.concatenate(samples)
    output['sample'] = sample
    return output



def create_fns(batch_size, R, Ds, seed, leakiness=0.1, lr=0.0002, scaler=1,
               var_x=1):

    alpha = T.Placeholder((1,), 'float32')
    x = T.Placeholder((Ds[0],), 'float32')
    X = T.Placeholder((batch_size, Ds[-1]), 'float32')

    signs = T.Placeholder((np.sum(Ds[1:-1]),), 'float32')
    SIGNS = T.Placeholder((R, np.sum(Ds[1:-1])), 'float32')
    
    m0 = T.Placeholder((batch_size, R), 'float32')
    m1 = T.Placeholder((batch_size, R, Ds[0]), 'float32')
    m2 = T.Placeholder((batch_size, R, Ds[0], Ds[0]), 'float32')

    Ws, vs = init_weights(Ds, seed, scaler)
    Ws = [T.Variable(w, name='W' + str(l)) for l, w in enumerate(Ws)]
    vs = [T.Variable(v, name='v' + str(l)) for l, v in enumerate(vs)]

    var_x = T.Variable(T.ones(Ds[-1]) * var_x)
    var_z = T.Variable(T.ones(Ds[0]))

    # create the placeholders
    Ws_ph = [T.Placeholder(w.shape, w.dtype) for w in Ws]
    vs_ph = [T.Placeholder(v.shape, v.dtype) for v in vs]
    var_x_ph = T.Placeholder(var_x.shape, var_x.dtype)

    ############################################################################
    # Compute the output of g(x)
    ############################################################################

    maps = [x]
    xsigns = []
    masks = []
    
    for w, v in zip(Ws[:-1], vs[:-1]):
        
        pre_activation = T.matmul(w, maps[-1]) + v
        xsigns.append(T.sign(pre_activation))
        masks.append(relu_mask(pre_activation, leakiness))
        maps.append(pre_activation * masks[-1])

    xsigns = T.concatenate(xsigns)
    maps.append(T.matmul(Ws[-1], maps[-1]) + vs[-1])

    ############################################################################
    # compute the masks and then the per layer affine mappings
    ############################################################################

    cumulative_units = np.cumsum([0] + Ds[1:])
    xqs = relu_mask([xsigns[None, cumulative_units[i]:cumulative_units[i + 1]]
                    for i in range(len(Ds) - 2)], leakiness)
    qs = relu_mask([signs[None, cumulative_units[i]:cumulative_units[i + 1]]
                    for i in range(len(Ds) - 2)], leakiness)
    Qs = relu_mask([SIGNS[:, cumulative_units[i]:cumulative_units[i + 1]]
                    for i in range(len(Ds) - 2)], leakiness)

    Axs, bxs = get_Abs(Ws, vs, xqs)
    Aqs, bqs = get_Abs(Ws, vs, qs)
    AQs, bQs = get_Abs(Ws, vs, Qs)

    all_bxs = T.hstack(bxs[:-1]).transpose()
    all_Axs = T.hstack(Axs[:-1])[0]

    all_bqs = T.hstack(bqs[:-1]).transpose()
    all_Aqs = T.hstack(Aqs[:-1])[0]

    x_inequalities = T.hstack([all_Axs, all_bxs]) * xsigns[:, None]
    q_inequalities = T.hstack([all_Aqs, all_bqs]) * signs[:, None]

    ############################################################################
    # loss (E-step NLL)
    ############################################################################

    Bm0 = T.einsum('nd,Nn->Nd', bQs[-1], m0)
    B2m0 = T.einsum('nd,Nn->Nd', bQs[-1] ** 2, m0)
    Am1 = T.einsum('nds,Nns->Nd', AQs[-1], m1)
    ABm1 = T.einsum('nds,nd,Nns->Nd', AQs[-1], bQs[-1], m1)
    Am2AT = T.diagonal(T.einsum('nds,Nnsc,npc->Ndp', AQs[-1], m2, AQs[-1]),
                        axis1=1, axis2=2)
    xAm1Bm0 = X * (Am1 + Bm0)

    M2diag = T.diagonal(m2.sum(1), axis1=1, axis2=2)
    
    prior = sum([T.sum(w**2) for w in Ws], 0.) / cov_W + sum([T.sum(v**2) for v in vs[:-1]], 0.) / cov_b
    loss = - 0.5 * (T.log(var_x).sum() + T.log(var_z).sum()\
            + (M2diag / var_z).sum(1)\
            + ((X ** 2 - 2 * xAm1Bm0 + B2m0 + Am2AT + 2 * ABm1) / var_x).sum(1))

    mean_loss = - (loss + 0.5 * prior).mean()
    adam = sj.optimizers.SGD(mean_loss, 0.001, params=Ws + vs)

    ############################################################################
    # update of var_x
    ############################################################################

    update_varx = (X ** 2 - 2 * xAm1Bm0 + B2m0 + Am2AT + 2 * ABm1).mean()\
                    * T.ones(Ds[-1])
    update_varz = M2diag.mean() * T.ones(Ds[0])

    ############################################################################
    # update for biases IT IS DONE FOR ISOTROPIC COVARIANCE MATRIX
    ############################################################################

    FQ = get_forward(Ws, Qs)
    update_vs = {}
    for i in range(len(vs)):
        
        if i < len(vs) - 1:
            # now we forward each bias to the x-space except the ith
            separated_bs = bQs[-1] - T.einsum('nds,s->nd', FQ[i], vs[i])
            # compute the residual and apply sigma
            residual = (X[:, None, :] - separated_bs) * m0[:, :, None]\
                                         - T.einsum('nds,Nns->Nnd', AQs[-1], m1)
            back_error = T.einsum('nds,nd->s', FQ[i], residual.mean(0))
            probed = FQ[i]
            whiten = T.einsum('ndc,nds,n->cs', FQ[i] , FQ[i], m0.mean(0))
            whiten = whiten + T.eye(whiten.shape[-1]) / cov_b
            update_vs[vs[i]] = T.linalg.solve(whiten, back_error)
        else:
            back_error = (X - (Am1 + Bm0) + vs[-1])
            update_vs[vs[i]] = back_error.mean(0)

    ############################################################################
    # update for slopes IT IS DONE FOR ISOTROPIC COVARIANCE MATRIX 
    ############################################################################

    update_Ws = {}
    for i in range(len(Ws)):
        
        U = T.einsum('nds,ndc->nsc', FQ[i], FQ[i])
        if i == 0:
            V = m2.mean(0)
        else:
            V1 = T.einsum('nd,nq,Nn->ndq', bQs[i-1], bQs[i-1], m0)
            V2 = T.einsum('nds,nqc,Nnsc->ndq', AQs[i-1], AQs[i-1], m2)
            V3 = T.einsum('nds,nq,Nns->ndq', AQs[i-1], bQs[i-1], m1)
            Q = T.einsum('nd,nq->ndq', Qs[i - 1], Qs[i - 1])
            V = Q * (V1 + V2 + V3 + V3.transpose((0, 2, 1))) / batch_size

        whiten = T.stack([T.kron(U[n], V[n]) for n in range(V.shape[0])]).sum(0)
        whiten = whiten + T.eye(whiten.shape[-1]) / cov_W
        # compute the residual (bottom up)
        if i == len(Ws) - 1:
            bottom_up = (X[:, None, :] - vs[-1])
        else:
            if i == 0:
                residual = (X[:, None, :] - bQs[-1])
            else:
                residual = (X[:, None, :] - bQs[-1]\
                            + T.einsum('nds,ns->nd', FQ[i - 1], bQs[i-1]))
            bottom_up = T.einsum('ndc,Nnd->Nnc', FQ[i], residual)

        # compute the top down vector
        if i == 0:
            top_down = m1
        else:
            top_down = Qs[i - 1] * (T.einsum('nds,Nns->Nnd', AQs[i - 1], m1) +\
                               T.einsum('nd,Nn->Nnd', bQs[i - 1], m0))

        vector = T.einsum('Nnc,Nns->cs', bottom_up, top_down) / batch_size
        condition = T.diagonal(whiten)
        update_Ws[Ws[i]] = T.linalg.solve(whiten, vector.reshape(-1)).reshape(Ws[i].shape)

    ############################################################################
    # create the io functions
    ############################################################################

    params = sj.function(outputs = Ws + vs + [var_x])
    ll = T.Placeholder((), 'int32')
    selector = T.one_hot(ll, len(vs))
    for i in range(len(vs)):
        update_vs[vs[i]] = ((1 - alpha) * vs[i] + alpha * update_vs[vs[i]])\
                            * selector[i] + vs[i] * (1 - selector[i])
    for i in range(len(Ws)):
        update_Ws[Ws[i]] = ((1 - alpha) * Ws[i] + alpha * update_Ws[Ws[i]])\
                            * selector[i] + Ws[i] * (1 - selector[i])

    output = {'train':sj.function(SIGNS, X, m0, m1, m2, outputs=mean_loss,
                                  updates=adam.updates),
              'update_var':sj.function(SIGNS, X, m0, m1, m2, outputs=mean_loss,
                                        updates = {var_x: update_varx}),
              'update_vs':sj.function(alpha, ll, SIGNS, X, m0, m1, m2, outputs=mean_loss,
                                      updates = update_vs),
              'loss':sj.function(SIGNS, X, m0, m1, m2, outputs=mean_loss),
              'update_Ws':sj.function(alpha, ll, SIGNS, X, m0, m1, m2, outputs=mean_loss,
                                      updates = update_Ws),
              'signs2Ab': sj.function(signs, outputs=[Aqs[-1][0], bqs[-1][0]]),
              'signs2ineq': sj.function(signs, outputs=q_inequalities),
              'g': sj.function(x, outputs=maps[-1]),
              'input2all': sj.function(x, outputs=[maps[-1], Axs[-1][0],
                                       bxs[-1][0], x_inequalities, xsigns]),
              'get_nll': sj.function(SIGNS, X, m0, m1, m2, outputs=mean_loss),
              'assign': sj.function(*Ws_ph, *vs_ph, var_x_ph,
                                    updates=dict(zip(Ws + vs + [var_x],
                                             Ws_ph + vs_ph + [var_x_ph]))),
              'varx': sj.function(outputs=var_x),
              'varz': sj.function(outputs=var_z),
              'params': params,
              'probed' : sj.function(SIGNS, X, m0, m1, m2, outputs=probed),
              'input2signs': sj.function(x, outputs=xsigns),
              'S' : Ds[0], 'D':  Ds[-1], 'R': R, 'model': 'EM', 'L':len(Ds)-1,
              'kwargs': {'batch_size': batch_size, 'Ds':Ds, 'seed':seed,
                    'leakiness':leakiness, 'lr':lr, 'scaler':scaler}}
 
    def sample(n):
        samples = []
        for i in range(n):
            samples.append(output['g'](np.random.randn(Ds[0])))
        return np.array(samples)
    
    output['sample'] = sample

    return output


def NLL(model, DATA):


    S, D, R = model['S'], model['D'], model['R']
    z = np.random.randn(S)/10

    m0 = np.zeros((DATA.shape[0], R))
    m1 = np.zeros((DATA.shape[0], R, S))
    m2 = np.zeros((DATA.shape[0], R, S, S))
    m_loss = []
    output, A, b, inequalities, signs = model['input2all'](z)
    regions = utils.search_region(model['signs2ineq'], model['signs2Ab'],
                                  signs)
    batch_signs = np.pad(np.array(list(regions.keys())),
                         [[0, R - len(regions)], [0, 0]])


    varx = np.eye(D) * model['varx']()
    varz = np.eye(S) * model['varz']()

    m0[:, :len(regions)], m1[:, :len(regions)], m2[:, :len(regions)] = utils.marginal_moments(DATA, regions, varx, varz)[1:]

    return model['loss'](batch_signs, DATA, m0, m1, m2)


def EM(model, DATA, epochs, n_iter, update_var=False, pretrain=False):

    batch_size = model['kwargs']['batch_size']
    if model['model'] == 'VAE':
        L = []
        for e in range(epochs):
            for i in range(len(DATA) // batch_size):
                L.append(model['train'](DATA[i * batch_size: (i + 1) * batch_size]))
        return L


    # PRETRAIN WITH GLO
    if pretrain:
        glo = create_glo(**model['kwargs'])
        error = 1
        cpt = 0
        Z = np.random.randn(DATA.shape[0], model['kwargs']['Ds'][0])
        while 1:
            cpt +=1
            II = np.random.permutation(len(DATA))[:batch_size]
            glo['reset'](Z[II])
            bat = DATA[II]
            for j in range(10):
                z = glo['estimate'](bat)
            Z[II]=z
            for j in range(4):
                glo['train'](bat)
            error = glo['loss'](bat)
            if cpt % 200 == 0:
                print(cpt, error)
            if cpt > 40000:
                break
     
        # THEN SET IT UP
        print('Setting pu the weights')
        model['assign'](*glo['params']())

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
#        others = utils.search_region_sample(model['input2signs'])
#        print('regions', len(regions), len(others))
#        print('Equal ?', regions.keys() == others)
#        print(regions.keys())
#        print(utils.search_region_sample(model['input2signs']))
        for r in regions:
            print(utils.get_vertices(regions[r]['ineq'][:, :-1], regions[r]['ineq'][:,-1]))
        if len(regions) > R:
            print('ALARMMM')
            print(model['params']())
        batch_signs = np.pad(np.array(list(regions.keys())),
                             [[0, R - len(regions)], [0, 0]])

        varx = np.eye(D) * model['varx']()
        varz = np.eye(S) * model['varz']()
        print('varx', np.diag(varx))
        print('varz', np.diag(varz))
   
        m0 *= 0
        m1 *= 0
        m2 *= 0
        m0[:, :len(regions)], m1[:, :len(regions)], m2[:, :len(regions)] = utils.marginal_moments(DATA, regions, varx, varz)[1:]

        m_loss.append(model['loss'](batch_signs, DATA, m0, m1, m2))
        print('after E step', m_loss[-1])

        for i in range(n_iter):
            if update_var:
                m_loss.append(model['update_var'](batch_signs, DATA, m0, m1, m2))
#            m_loss.append(model['train'](batch_signs, DATA, m0, m1, m2))
            if i %10 == 0:
                params = model['params']()
                print('here?',np.max(params[0]), np.max(params[1]), np.max(params[2]), np.max(params[3]),
                        model['loss'](batch_signs, DATA, m0, m1, m2))
            for l in np.random.permutation(model['L']):#-1, -1, -1):
                if np.random.randn() < 0:
                    m_loss.append(model['update_vs'](0.05, l, batch_signs, DATA, m0, m1, m2))
                else:
                    m_loss.append(model['update_Ws'](0.05, l, batch_signs, DATA, m0, m1, m2))
        print('after M step', model['loss'](batch_signs, DATA, m0, m1, m2))
        if n_iter > 1:
            print('strictly decreasing M step ?:', np.diff(m_loss).max())
    return m_loss



