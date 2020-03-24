import sys
sys.path.insert(0, "../SymJAX")
import numpy as np
import symjax as sj
import symjax.tensor as T
import cdd
import matplotlib.pyplot as plt
import utils



input = T.Placeholder((10, 1), 'float32')

layer1 = sj.layers.Dense(input, 32)
layer1relu = T.relu(layer1)

layer2 = sj.layers.Dense(layer1relu, 32)
layer2relu = T.relu(layer2)

output = sj.layers.Dense(layer2relu, 2)

signs = T.sign(T.hstack((layer1, layer2)))
A = sj.jacobian(output, [input])[0]
b = output - T.matmul(A, input)
H = T.hstack([sj.jacobian(layer1, [input])[0],
              sj.jacobian(layer2, [input])[0]])


f = sj.function(input, outputs=[A,b, signs, H])

f(np.random.randn((10, 1)))
assdf
# toy example
#m = cdd.Matrix([[0, 1., 0], [2, -1., 0], [2, 0., -1.], [0, 0, 1.]])
#m = cdd.Matrix([[1, -1., 1], [1, 1., -1], [1, -1., -1.], [1, 1, 1.]])
m = cdd.Matrix([[20, 1., 0], [20, -1., 0], [20, 0., -1.], [20, 0, 1.]])
m = cdd.Matrix([[200, 1., 0, 0], [200, -1., 0, 0], [200, 0., -1., 0],
                [200, 0, 1., 0], [200., 0., 0, 1], [200, 0, 0, -1]])
m.rep_type = cdd.RepType.INEQUALITY
p = cdd.Polyhedron(m)
v = np.array(p.get_generators())[:, 1:]

accR = 0.
V = 0.
mu = np.ones(3) * 0
cov = 2*np.eye(3) + (np.eye(3)*0.1)[:, ::-1]


for simplex in utils.get_simplices(v):
    a, s = utils.simplex_to_cones(v[simplex])
    for (A, b), ss in zip(a, s):
        lower, mu_, cov_, R = utils.planes_to_rectangle(A, b, mu, cov)
        invR = np.linalg.inv(R)
        value = invR.dot(utils.E_Y(lower, mu_, cov_))
        value_ = invR.dot(utils.E_YYT(lower, mu_, cov_).dot(invR.T))
        value2 = utils.E_1(lower - mu_, cov_)
        accR += ss * value_
        V += ss * value2

print('R', accR/V)
