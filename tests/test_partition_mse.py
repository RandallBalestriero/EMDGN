import sys
sys.path.insert(0, "../SymJAX")
import numpy as np
import symjax as sj
import symjax.tensor as T
import cdd
import utils
from tqdm import tqdm
from scipy.spatial import ConvexHull
from multiprocessing import Pool
import tabulate

def create_fns(input, in_signs, Ds):

    cumulative_units = np.concatenate([[0], np.cumsum(Ds[:-1])])
    
    Ws = [sj.initializers.he((j, i)) for j, i in zip(Ds[1:], Ds[:-1])]
    bs = [sj.initializers.he((j,)) for j in Ds[1:]]

    A_w = [T.eye(Ds[0])]
    B_w = [T.zeros(Ds[0])]
    
    A_q = [T.eye(Ds[0])]
    B_q = [T.zeros(Ds[0])]
    
    maps = [input]
    signs = []
    masks = [T.ones(Ds[0])]
    in_masks = T.where(T.concatenate([T.ones(Ds[0]), in_signs]) > 0, 1.,
                                     0.1)

    for w, b in zip(Ws[:-1], bs[:-1]):
        
        pre_activation = T.matmul(w, maps[-1]) + b
        signs.append(T.sign(pre_activation))
        masks.append(T.where(pre_activation > 0, 1., 0.1))

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

error1, error2 = [], []
for k in range(2, 5):
    for Ds in [[k, 4, 1], [k, 4, 3], [k, 8, 1], [k, 8, 3],
                [k, 4, 4, 3, 1], [k, 4, 4, 3, 3]]:
        print(Ds)
        input = T.Placeholder((Ds[0],), 'float32')
        in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')
        
        f, g, h, all_g = create_fns(input, in_signs, Ds)
        
        x = np.random.randn(Ds[0])/10
        output, A, b, inequalities, signs = f(x)
        regions = []
        utils.search_region(all_g, regions, signs)
        
        if Ds[0] == 2:
            K=100
            xx = np.meshgrid(np.linspace(-10, 10, K), np.linspace(-10, 10, K))
            xx = np.vstack([xx[0].flatten(), xx[1].flatten()]).T
        elif Ds[0] == 3:
            K=60
            xx = np.meshgrid(np.linspace(-10, 10, K), np.linspace(-10, 10, K), np.linspace(-10, 10, K))
            xx = np.vstack([xx[0].flatten(), xx[1].flatten(), xx[2].flatten()]).T
        else:
            K=20
            xx = np.meshgrid(np.linspace(-10, 10, K), np.linspace(-10, 10, K), np.linspace(-10, 10, K), np.linspace(-10, 10, K))
            xx = np.vstack([xx[0].flatten(), xx[1].flatten(), xx[2].flatten(), xx[3].flatten()]).T
        
        yy = np.zeros((int(K ** Ds[0]), Ds[-1]))
        yy2 = np.zeros((int(K ** Ds[0]), Ds[-1]))

        
        allA, allB = [], []
        
        for flips in regions:
            
            A_w, b_w = g(flips)
            allA.append(A_w)
            allB.append(b_w)
        
        allA = np.array(allA)
        allB = np.array(allB)
        
        flips = np.array(regions)
        
        w = utils.find_region(xx, flips, f)
        
        for i in range(len(regions)):
            yy[w == i] = np.dot(xx[w == i], allA[i].T) + allB[i]
        
        for k, x in enumerate(xx):
            yy2[k] = h(x)
        
        error1.append(np.abs(yy - yy2).mean())
        error2.append(((yy - yy2)**2).mean())


print(np.array(error1).reshape((3, 6)))
print(np.array(error2).reshape((3, 6)))

sadf




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
for x in tqdm(xx[[0]]):
    p.append(utils.algo2(x, ineq_A, ineq_B, signs, regions,sigma_x, mu_z, sigma_z, g)[0])

asdf
plt.subplot(121)
plt.imshow(np.array(p).reshape(50, 50), aspect='auto')

plt.show()
print(p)
