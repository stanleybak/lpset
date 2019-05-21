'''
Tests for LP operations. Made for use with py.test:

> python3 -m pytest test_*.py
'''

import random
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import swiglpk as glpk

import lpset
import lpplot
from util import assert_verts_is_box, assert_verts_equals
import kamenev

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

def test_verts_degenerate():
    'test constructing verts with degenerate polytopes'

    # list of tests to skip (for isolating tests when debugging)
    skip_tests = []# [0, 1, 2, 3, 4, 5]

    # 2-d box (non-degenerate)
    if 0 not in skip_tests:
        m = lpset.from_box([[0, 1], [0, 1]])
        verts = lpplot.get_verts_nd(m.lpi, m.lpi.dims)
        assert_verts_equals(verts, [[0, 0], [0, 1], [1, 1], [1, 0]])

    # 4-d box with two of the dimensions being flat
    if 1 not in skip_tests:
        m = lpset.from_box([[5, 5], [1, 2], [7, 7], [3, 5]])
        verts = lpplot.get_verts_nd(m.lpi, m.lpi.dims)
        assert_verts_equals(verts, [[5, 1, 7, 3], [5, 2, 7, 3], [5, 2, 7, 5], [5, 1, 7, 5]])

    if 2 not in skip_tests:
        # unit square with constraints x+y <= 0 (should just be the origin)
        mat = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]]
        rhs = [1, 0, 1, 0, 0]
        m = lpset.LpSet(mat, rhs)
        verts = lpplot.get_verts_nd(m.lpi, m.lpi.dims)
        assert_verts_equals(verts, [0, 0])

    if 3 not in skip_tests:
        # unit square with constraints x+y = 2 (should just be the [1, 1])
        mat = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]]
        rhs = [1, 0, 1, 0, 2, -2]
        m = lpset.LpSet(mat, rhs)
        verts = lpplot.get_verts_nd(m.lpi, m.lpi.dims)
        assert_verts_equals(verts, [1, 1])

    if 4 not in skip_tests:
        # unit square with constraints x+y = 1
        mat = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]]
        rhs = [1, 0, 1, 0, 1, -1]
        m = lpset.LpSet(mat, rhs)
        verts = lpplot.get_verts_nd(m.lpi, m.lpi.dims)
        assert_verts_equals(verts, [[1, 0], [0, 1]])

    if 5 not in skip_tests:
        # unit cube with constraints x+y+z = 1
        # set is a 2-d triangle in 3-d space
        mat = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1], [-1, -1, -1]]
        rhs = [1, 0, 1, 0, 1, 0, 1, -1]
        m = lpset.LpSet(mat, rhs)
        verts = lpplot.get_verts_nd(m.lpi, m.lpi.dims)
        assert_verts_equals(verts, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # 2-d box in 3-d
    if 6 not in skip_tests:
        m = lpset.from_box([[0, 1], [0, 1], [5, 5]])
        verts = lpplot.get_verts_nd(m.lpi, m.lpi.dims)
        assert_verts_equals(verts, [[0, 0, 5], [0, 1, 5], [1, 1, 5], [1, 0, 5]])

    if 7 not in skip_tests:
        # unit 4-d cube with constraints w+x+y+z = 1
        # set is a 3-d simplex in 4-d space
        mat = [[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0],
               [0, 0, 0, 1], [0, 0, 0, -1], [1, 1, 1, 1], [-1, -1, -1, -1]]
        rhs = [1, 0, 1, 0, 1, 0, 1, 0, 1, -1]
        m = lpset.LpSet(mat, rhs)
        verts = lpplot.get_verts_nd(m.lpi, m.lpi.dims)
        assert_verts_equals(verts, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    assert not skip_tests, "failing because some tests were skipped"

def test_matthias_zonotopes():
    'tests from_centered_zonotope constructor'

    # zonotopes from Figure 2 of "On Computing the Minkowski Difference of Zonotopes" by Matthias Althoff

    # Z_m (shifted due to lack of center in our constructor)
    if False:
        m = lpset.from_centered_zonotope([[1, 0], [0, 1], [1, 1]])
        expected = [[-2, -2], [-2, 0], [0, 2], [2, 2], [2, 0], [0, -2]]
        assert_verts_equals(m.verts(), expected)


        # Z_{s,1}
        s1 = lpset.from_centered_zonotope([[0.5, -0.25], [0, 0.25]])
        expected = [[-0.5, 0], [-0.5, 0.5], [0.5, 0], [0.5, -0.5]]
        assert_verts_equals(s1.verts(), expected)

    # Z_{s,2}
    s2 = lpset.from_centered_zonotope([[0.5, -0.5], [0, 0.5]])

    #s2.plot()
    #plt.show()
    
    expected = [[-0.5, 0], [-0.5, 1.0], [0.5, 0], [0.5, -1.0]]
    assert_verts_equals(s2.verts(), expected)

    if False:
        # Z_{s,3}
        s3 = lpset.from_centered_zonotope([[2.0, -0.5], [0, 0.5]])
        expected = [[-2.0, 0], [-2.0, 1.0], [2.0, 0], [2.0, -1.0]]
        assert_verts_equals(s3.verts(), expected)

def test_3d_box_plot():
    'test verts3d on unit box'

    m = lpset.from_box([[0, 1], [0, 1], [0, 1]])
    
    verts = m.verts3d()
    assert len(verts) == 8

    # figure should work
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    m.plot3d(ax)

    #plt.show()

print("debug 3dboxplot test")
test_3d_box_plot()

def test_3d_zonotope_plot():
    'test verts3d on zonotope'

    # zonotope from Figure 6 of "On Computing the Minkowski Difference of Zonotopes" by Matthias Althoff

    m = lpset.from_centered_zonotope([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    #s = lpset.from_zonotope([0, 0, 0], [[-1/3, 1/3, 1/3], [1/3, 0, 0], [0, 1/3, 0], [0, 0, 1/3]])

    verts = m.verts3d()
    assert len(verts) == 14

    # figure should work
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    m.plot3d(ax)

    plt.show()

    assert False

def test_complement_lps():
    'test getting the complement lps'


    m = lpset.from_centered_zonotope([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    c = m.get_complement_lpsets()

    assert len(c) == 8

def test_mindiff_2d():
    'test minkowski difference in 2d'

    # Z_m (shifted due to lack of center in our constructor)
    m = lpset.from_centered_zonotope([[1, 0], [0, 1], [1, 1]])

    # Z_{s,1}
    s1 = lpset.from_centered_zonotope([[0.5, -0.25], [0, 0.25]])

    p_in = m.find_point_in_diff(s1)

    print(p_in)

    assert False

def test_mindiff_3d():
    'test minkowski difference in 3d'

    m = lpset.from_centered_zonotope([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    s = lpset.from_centered_zonotope([[-1/3, 1/3, 1/3], [1/3, 0, 0], [0, 1/3, 0], [0, 0, 1/3]])

    p_in = m.find_point_in_diff(s)

    print(p_in)

    assert False

def test_intersection():
    'test the lpset intersection'

    m = lpset.from_centered_zonotope([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    s = lpset.from_centered_zonotope([[-1/3, 1/3, 1/3], [1/3, 0, 0], [0, 1/3, 0], [0, 0, 1/3]])

    inter = lpset.intersection(m, s)

    verts_s = s.verts3d()
    verts_inter = lpplot.get_verts3d(inter)

    assert_verts_equals(verts_s, verts_inter)

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

def test_hyperplane2d():
    'test constructing the normal of the hyperplane through some points'

    pts = [[1, 1], [2, 1]]

    normal, rhs = kamenev.hyperplane(pts, [0, 0])
    
    # result: y = 1
    assert np.allclose(normal, [0, 1])
    assert np.allclose(1, [rhs])

    normal, rhs = kamenev.hyperplane(pts, [0, 3])
    
    # result: -y = -1
    assert np.allclose(normal, [0, -1])
    assert np.allclose(-1, [rhs])

def test_hyperplane3d():
    'test hyperplane construction in 3d'

    pts = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    normal, rhs = kamenev.hyperplane(pts, [0.5, 0.5, 0.5])

    for pt in pts:
        assert np.allclose([np.dot(pt, normal)], [rhs])

def test_regular_simplex_vecs():
    'tests for n dimensional regular simplex'

    for dims in [2, 3, 5, 10]:

        pts = kamenev.regular_simplex_vecs(dims)
        assert len(pts) == dims + 1

        # compute centroid
        centroid_sum = [0] * dims

        for pt in pts:
            assert len(pt) == dims

            for index, x in enumerate(pt):
                centroid_sum[index] += x

        for index in range(dims):
            centroid_sum[index] /= len(pts)

        assert np.allclose(np.array(centroid_sum), np.zeros((dims,))), "centroid was not zero"

        # make sure each vec is equidistant from the other ones
        random.seed(0)
        num_samples = 10

        for _ in range(num_samples):
            # pick two random edges

            index_a = random.randint(0, dims)
            index_b = random.randint(0, dims - 1)

            # make sure we don't pick the same index
            if index_b >= index_a:
                index_b += 1

            a = np.array(pts[index_a], dtype=float)
            b = np.array(pts[index_b], dtype=float)

            index_c = random.randint(0, dims)
            index_d = random.randint(0, dims - 1)

            # make sure we don't pick the same index
            if index_d >= index_c:
                index_d += 1

            c = np.array(pts[index_c], dtype=float)
            d = np.array(pts[index_d], dtype=float)

            dist_ab = np.linalg.norm(a - b)
            dist_cd = np.linalg.norm(c - d)
            tol = 1e-9

            assert abs(dist_ab - dist_cd) < tol, f"distance between points {index_a} and {index_b} ({dist_ab}) is " + \
                f"different distance between points {index_c} and {index_d} ({dist_cd})"
