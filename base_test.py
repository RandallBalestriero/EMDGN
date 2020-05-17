import numpy as np
import cdd
import matplotlib.pyplot as plt
import utils


mu = np.zeros(2)+4.3
cov = np.array([[2.1, 0.3],[0.3, 0.8]])


m = cdd.Matrix([[30, 1., 0], [30, 0., 1.1]])
#print(utils.get_vertices(np.array(m)))
m.rep_type = cdd.RepType.INEQUALITY
print(utils.phis_w(np.array(m), mu, cov), np.outer(mu, mu)+cov)
print('\n\n\n')

m = cdd.Matrix([[40, 0., 1], [40, -1., 0], [40, 1., -1.]])
#print(utils.get_vertices(np.array(m)))
m.rep_type = cdd.RepType.INEQUALITY
print(utils.phis_w(np.array(m), mu, cov))
print('\n\n\n')

m = cdd.Matrix([[40, 1., 0], [40, -1., 0], [40, 0., -1.], [40, 0, 1.]])
#print(utils.get_vertices(np.array(m)))
m.rep_type = cdd.RepType.INEQUALITY
print(utils.phis_w(np.array(m), mu, cov))
print('\n\n\n')

m = cdd.Matrix([[40, 1., 0], [40, -1., 0], [40, 0., -1.], [40, 0, 1.],
                [40, 1., 1], [40, -1, -1]])
print('v',utils.get_vertices(np.array(m)))
m.rep_type = cdd.RepType.INEQUALITY
print('p',utils.phis_w(np.array(m), mu, cov))
print('\n\n\n')


