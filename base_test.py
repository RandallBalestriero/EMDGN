import numpy as np
import cdd
import matplotlib.pyplot as plt
import utils

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
