'''
Tests for LP operations. Made for use with py.test:

> python3 -m pytest test_*.py
'''

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import swiglpk as glpk

import lpset
from util import assert_verts_is_box, assert_verts_equals

def test_from_box():
    'tests from_box constructor'

    l = [[-5, -4], [0, 1]]
    box = lpset.from_box(l)

    assert_verts_is_box(box.verts(), l)

def test_2d_zonotope():
    'simple zonotope test'

    m = lpset.from_zonotope([2, 3], [[1, 1], [0, 1]])

    #m.plot()
    #plt.show()
    
    #expected = [[-1, -2], [-1, 0], [1, 2], [1, 0]]
    #assert_verts_equals(m.verts(), expected)

    mat = np.array([[ 0.,  0.],
       [-1.,  0.],
       [-1.,  1.],
       [ 1.,  0.],
       [ 0.,  0.],
       [ 0.,  1.],
       [ 1., -1.],
       [ 0., -1.],
       [ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]])
    
    rhs = np.array([ 2.,  0.,  2.,  4.,  2.,  4.,  0., -2.,  0.,  2.,  0.])

    m2 = lpset.LpSet(mat, rhs)

    m.plot('k')
    m2.plot('r--')
    plt.show()

def test_mattias_zonotope():
    'tests from_zonotope constructor'

    # zonotopes from Figure 2 of "On Computing the Minkowski Difference of Zonotopes" by Matthias Althoff

    # Z_m
    m = lpset.from_zonotope([1, 1], [[1, 0], [0, 1], [1, 1]])
    expected = [[-1, -1], [-1, 1], [1, 3], [3, 3], [3, 1], [1, -1]]
    assert_verts_equals(m.verts(), expected)

    # Z_{s,1}
    s1 = lpset.from_zonotope([0, 0], [[0.5, -0.25], [0, 0.25]])
    expected = [[-0.5, 0], [-0.5, 0.5], [0.5, 0], [0.5, -0.5]]
    assert_verts_equals(s1.verts(), expected)

    # Z_{s,2}
    s2 = lpset.from_zonotope([0, 0], [[0.5, -0.5], [0, 0.5]])
    expected = [[-0.5, 0], [-0.5, 1.0], [0.5, 0], [0.5, -1.0]]
    assert_verts_equals(s2.verts(), expected)

    # Z_{s,3}
    s3 = lpset.from_zonotope([0, 0], [[2.0, -0.5], [0, 0.5]])
    expected = [[-2.0, 0], [-2.0, 1.0], [2.0, 0], [2.0, -1.0]]
    assert_verts_equals(s3.verts(), expected)

def test_3d_zonotope():
    '3d zonotope'

    m = lpset.from_zonotope([0, 0, 0], [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    #s = lpset.from_zonotope([0, 0, 0], [[-1/3, 1/3, 1/3], [1/3, 0, 0], [0, 1/3, 0], [0, 0, 1/3]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    m.plot3d(ax)


    plt.show()
    
def test_minkowski_difference():
    'test minkowski difference'

    # tests from Figure 2 of "On Computing the Minkowski Difference of Zonotopes" by Matthias Althoff

    # Z_m
    z_m = lpset.from_zonotope([1.1, 1.2], [[1.3, 0.1], [0.2, 1.4], [1.5, 1.6]])

    print(z_m)
    assert False

    # Z_{s,1}
    z_s1 = lpset.from_zonotope([0, 0], [[0.5, -0.25], [0, 0.25]])

    z_d1 = z_m.clone().minkowski_difference(z_s1)

    z_m.plot(':')
    z_s1.plot('-')
    z_d1.plot('--')

    plt.show()
