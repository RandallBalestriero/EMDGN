import sys
sys.path.insert(0, "../SymJAX")
import numpy as np
import symjax as sj
import symjax.tensor as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import utils
import networks
from tqdm import tqdm



for Ds in [[2, 4, 1], [2, 8, 1], [2, 3, 3, 2, 1]]:
    input = T.Placeholder((Ds[0],), 'float32')
    in_signs = T.Placeholder((np.sum(Ds[1:-1]),), 'bool')
    
    f, g, h, all_g = create_fns(input, in_signs, Ds)
    
    x = np.random.randn(Ds[0])/10
    output, A, b, inequalities, signs = f(x)
    regions = {}
    utils.search_region(all_g, regions, signs)
    
    K=200
    xx = np.meshgrid(np.linspace(-10, 10, K), np.linspace(-10, 10, K))
    xx = np.vstack([xx[0].flatten(), xx[1].flatten()]).T
    
    
    yy = np.zeros((K * K, 1))
    yy2 = np.zeros((K * K, 1))
    
    allA, allB, ineq_A, ineq_B = [], [], [], []
    
    for flips, ineq in regions.items():
        m = cdd.Matrix(np.hstack([ineq[:, [0]], ineq[:, 1:]]))
        m.rep_type = cdd.RepType.INEQUALITY
        v = np.array(cdd.Polyhedron(m).get_generators())[:, 1:]
        if len(v) > len(x):
            for simplex in get_simplices(v):
                A_w, b_w = g(flips)
                allA.append(A_w)
                allB.append(b_w)
                m = cdd.Matrix(v[simplex]))
        m.rep_type = cdd.RepType.INEQUALITY
        v = np.array(cdd.Polyhedron(m).get_generators())[:, 1:]
                
        else:
            A_w, b_w = g(flips)
            allA.append(A_w)
            allB.append(b_w)
            ineq_A.append(ineq[:, 1:])
            ineq_B.append(ineq[:, 0])
 
    
    allA = np.array(allA)
    allB = np.array(allB)
    
    flips = np.array(regions)
    
    w = utils.find_region(xx, flips, f)
    
    for i in range(len(regions)):
        yy[w == i] = np.dot(xx[w == i], allA[i].T) + allB[i]
    
    for k, x in enumerate(xx):
        yy2[k] = h(x)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(141)
    plt.imshow(w.reshape(K, K), aspect='auto', cmap='prism')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(142)
    plt.imshow(yy.reshape(K, K), aspect='auto')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
   
    plt.subplot(143)
    plt.imshow(yy2.reshape(K, K), aspect='auto')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
     
    plt.subplot(144)
    plt.imshow(np.abs(yy - yy2).reshape(K, K), aspect='auto')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])


    plt.savefig('error_prediction_{}.pdf'.format(Ds).replace(', ', '').replace('[', '').replace(']', ''))
    plt.close()

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
