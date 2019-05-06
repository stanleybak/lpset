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

    m = lpset.from_centered_zonotope([[1, 1], [0, 1]])

    m.plot()
    plt.show()
    
    expected = [[-1, -2], [-1, 0], [1, 2], [1, 0]]
    assert_verts_equals(m.verts(), expected)

def test_mattias_zonotopes():
    'tests from_centered_zonotope constructor'

    # zonotopes from Figure 2 of "On Computing the Minkowski Difference of Zonotopes" by Matthias Althoff

    # Z_m (shifted due to lack of center in our constructor)
    m = lpset.from_centered_zonotope([[1, 0], [0, 1], [1, 1]])
    expected = [[-2, -2], [-2, 0], [0, 2], [2, 2], [2, 0], [0, -2]]
    assert_verts_equals(m.verts(), expected)

    # Z_{s,1}
    s1 = lpset.from_centered_zonotope([[0.5, -0.25], [0, 0.25]])
    expected = [[-0.5, 0], [-0.5, 0.5], [0.5, 0], [0.5, -0.5]]
    assert_verts_equals(s1.verts(), expected)

    # Z_{s,2}
    s2 = lpset.from_centered_zonotope([[0.5, -0.5], [0, 0.5]])
    expected = [[-0.5, 0], [-0.5, 1.0], [0.5, 0], [0.5, -1.0]]
    assert_verts_equals(s2.verts(), expected)

    # Z_{s,3}
    s3 = lpset.from_centered_zonotope([[2.0, -0.5], [0, 0.5]])
    expected = [[-2.0, 0], [-2.0, 1.0], [2.0, 0], [2.0, -1.0]]
    assert_verts_equals(s3.verts(), expected)

def test_3d_zonotope():
    '3d zonotope'

    # zonotope from Figure 6 of "On Computing the Minkowski Difference of Zonotopes" by Matthias Althoff

    m = lpset.from_centered_zonotope([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    #s = lpset.from_zonotope([0, 0, 0], [[-1/3, 1/3, 1/3], [1/3, 0, 0], [0, 1/3, 0], [0, 0, 1/3]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    m.plot3d(ax)


    plt.show()

def test_simple_minkowski_difference():
    'test minkowski difference on box sets in 1 and 2d'

    a = lpset.from_box([[-10, 10]])
    b = lpset.from_box([[-1, 1]])
    dif = a.clone().minkowski_difference(b)

    # should be the interval [-9, 9]
    assert not lpset.contains_point(a, [-9.1])
    assert not lpset.contains_point(a, [9.1])
    assert lpset.contains_point(a, [-8.9])
    assert lpset.contains_point(a, [8.9])

    a = lpset.from_box([[-1, 1], [-1, 1]])
    b = lpset.from_box([[-0.2, 0.2], [-0.2, 0.2]])
    dif = a.clone().minkowski_difference(b)

    expected = [[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]]
    assert_verts_equals(dif.verts(), expected)

def test_chull_easy():
    'test convex hull operation'

    a = lpset.from_box([[10, 11], [10, 11]])
    b = lpset.from_box([[15, 17], [15, 17]])

    res = lpset.from_chull(a, b)

    print(res)

    c = lpset.from_forced_chull(a, b, lam=0.0)

    a.plot('r-')
    b.plot('b-')

    c.plot('g--')
    res.plot('k:')
    plt.show()

    assert False

def test_chull_hard():
    'test convex hull operation'

    m = lpset.from_centered_zonotope([[1, 0], [0, 1], [1, 1]])

    # Z_{s,1}
    s1 = lpset.from_centered_zonotope([[0.5, -0.25], [0, 0.25]])

    # Z_{s,2}
    s2 = lpset.from_centered_zonotope([[0.5, -0.5], [0, 0.5]])

    # s = s1
    s = s2


    #combined = lpset.minkowski_sum([m, s1])
    combined = lpset.minkowski_sum([m, s])

    lam = -0.1
    slack = 0.1
    c = lpset.from_forced_chull(m, combined, lam=lam, slack=slack)

    a_index = c.lpi.names.index('a')

    direction_vec = [0] * a_index + [1]
    c.lpi.set_minimize_direction(direction_vec, offset=0)
    val = c.lpi.minimize(columns=[a_index])[0]
    
    print(f"min_val of a: {val}")

    direction_vec = [0] * a_index + [-1]
    c.lpi.set_minimize_direction(direction_vec, offset=0)
    val = c.lpi.minimize(columns=[a_index])[0]

    print(f"max_val of a: {val}")

    m.plot('b:')
    s.plot('r:')
    combined.plot('k--')
    c.plot('g-')
    plt.show()

    assert False

def test_minkowski_dif_twogen():
    'test the minkowski difference from a zonotope with two linearly dependent generators'

    z = lpset.from_centered_zonotope([[0, 1], [0, -1]])
    expected = [[-2, 0], [2, 0]]
    assert_verts_equals(z.verts(), expected)

    b = lpset.from_box([[-0.1, 0.1]])
    dif = z.clone().minkowski_difference(b)

    expected = [[-1.9, 0], [1.9, 0]]
    assert_verts_equals(dif.verts(), expected)

def test_minkowski_difference():
    'test minkowski difference'

    # tests from Figure 2 of "On Computing the Minkowski Difference of Zonotopes" by Matthias Althoff

    # Z_m (shifted due to lack of center in our constructor)
    z_m = lpset.from_centered_zonotope([[1, 0], [0, 1], [1, 1]])

    # Z_{s,1}
    z_s1 = lpset.from_centered_zonotope([[0.5, -0.25], [0, 0.25]])

    z_d1 = z_m.clone().minkowski_difference(z_s1)

    z_m.plot(':')
    z_s1.plot('-')
    z_d1.plot('--')

    plt.show()
