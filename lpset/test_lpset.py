'''
Tests for LP operations. Made for use with py.test:

> python3 -m pytest test_*.py
'''

import matplotlib.pyplot as plt

import lpset
from util import assert_verts_is_box, assert_verts_equals

def test_from_box():
    'tests from_box constructor'

    l = [[-5, -4], [0, 1]]
    box = lpset.from_box(l)

    assert_verts_is_box(box.verts(), l)

def test_from_zonotope():
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
