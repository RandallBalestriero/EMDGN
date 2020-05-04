import sys
sys.path.insert(0, "../../SymJAX")
sys.path.insert(0, "../")
import numpy as np
import symjax as sj
import symjax.tensor as T
import utils
import networks
from tqdm import tqdm

error1, error2 = [], []
for Ds in [[1, 8, 8]]:
    print(Ds)
    model = networks.create_fns(1, 1, Ds, 0, var_x=np.ones(Ds[-1]))
    
    x = np.random.randn(Ds[0])/10
    output, A, b, inequalities, signs = model['input2all'](x)
    regions = utils.search_region(model['signs2ineq'], model['signs2Ab'],
                                  signs)
    print(len(regions))    
    K = 100
    xx = [np.linspace(-10, 10, K)] * Ds[0]
    xx = np.meshgrid(*xx)
    xx = np.vstack([x.flatten() for x in xx]).T
    
    yy = np.zeros((int(K ** Ds[0]), Ds[-1]))
    yy2 = np.zeros((int(K ** Ds[0]), Ds[-1]))

    
    As = np.array([regions[s]['Ab'][0] for s in regions])
    Bs = np.array([regions[s]['Ab'][1] for s in regions])

    
    w = utils.find_region(xx, regions, model['input2signs'])
    
    for i in range(len(regions)):
        yy[w == i] = np.dot(xx[w == i], As[i].T) + Bs[i]
    
    for k, x in enumerate(xx):
        yy2[k] = model['g'](x)
    
    error1.append(np.abs(yy - yy2).mean())
    error2.append(((yy - yy2)**2).mean())
    print(error1, error2)

print(np.array(error1).reshape((3, 6)))
print(np.array(error2).reshape((3, 6)))
