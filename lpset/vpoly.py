'''
Vertex polytope and related

Stanley Bak
'''

import numpy as np
from scipy.sparse import csr_matrix

from util import Freezable
from lpinstance import LpInstance

class VPoly(Freezable):
    '''
    polytope that's a convex hull of a list of vertices

    this one eliminates redundant vertices upon construction
    '''

    def __init__(self, pts, skip_remove_redundant=False):

        if not skip_remove_redundant:
            pts = remove_redundant(pts)

        self.lpi = make_verts_lpi(pts)

        self.freeze_attrs()

def remove_redundant(pts):
    'remove pts which are not on the chull'

    e = [pts[0]] # start with a single point

    lpi = make_verts_lpi(e)

    for pt in pts[1:]:
        # check if pt is in the lpi already

        if not pt_in_lpi(pt, lpi):
            vec = find_witness_vector(pt, e)

            # find the point in S which maximizes direction vec
            max_pt = pts[0]
            max_val = np.dot(vec, max_pt)

            for p in pts[1:]:
                val = np.dot(vec, p)

                if val > max_val:
                    max_val = val
                    max_pt = p
            
            add_chull_pt_to_lpi(max_pt, lpi)
            e.append(max_pt)

    return e

def find_witness_vector(pt, e_list):
    '''helper method for remove_redundant()

    find a witness vector n such that dot(n, e) < dot(n, pt), for all e in e_list
    '''

    dims = len(pt)

    # maximize dot(n, pt), subject to dot(n, e) - dot(n, pt) <= 0
    lpi = LpInstance()
    lpi.dims = dims
    lpi.add_rows_less_equal([0] * len(e_list))

    lpi.add_cols([f"n{i}" for i in range(dims)])

    # add constraints
    data = []
    inds = []
    indptr = [0]

    for e in e_list:
        for d in range(dims):
            data.append(e[d] - pt[d])
            inds.append(d)

        indptr.append(len(data))

    csr = csr_matrix((data, inds, indptr), shape=(len(e_list), dims), dtype=float)
    csr.check_format()

    lpi.set_constraints_csr(csr)

    # maximize dot(n, pt)
    lpi.set_minimize_direction(-1 * pt)

    cols = lpi.minimize()

    if cols is not None:
        # assignment exists, check that the constriants are all strict inequalities
        val = np.dot(cols, pt)
        raise RuntimeError('unimplemented')

def add_chull_pt_to_lpi(pt, lpi):
    'helper method for remove_redundant()'

    raise RuntimeError('unimplemented')

def pt_in_lpi(pt, lpi):
    'helper method for remove_redundant()'

    dims = len(pt)

    if lpi.get_num_rows() == dims + 1:
        # add rows for equality constraint
        lpi.add_rows_equal_zero(dims)

        # add constraint
        data = []
        inds = []
        indptr = [0]

        for d in dims:
            data.append(-1)
            inds.append(d)
            
            indptr.append(len(data))

        csr = csr_matrix((data, inds, indptr), shape=(dims, dims), dtype=float)
        csr.check_format()

        lpi.set_constraints_csr(csr, offset=(dims+1, 0))

    assert lpi.get_num_rows() == 2*dims + 1

    # update test point
    for d in range(dims):
        lpi.set_row_equal(dims + 1 + d, pt[d])

    return lpi.is_feasible()

def make_verts_lpi(pts):
    '''make an lpi that's a linear combination of the passed-in points, with a new variable for each point'''

    dims = len(pts[0])

    lpi = LpInstance()
    lpi.dims = dims
    lpi.add_rows_equal_zero(dims)
    lpi.add_rows_equal([1.0]) # lambda sum constraint row

    lpi.add_cols([f"x{i}" for i in range(dims)])
    lpi.add_double_bounded_cols([f"l{i}" for i in range(dims)], 0, 1) # 0 <= lambda <= 1

    # make constraints
    data = []
    inds = []
    indptr = [0]

    for d in dims:
        data.append(-1)
        inds.append(d)

        # add lambda value for each point
        for pt_index, pt in enumerate(pts):
            data.append(pt[d])
            inds.append(dims + pt_index)

        indptr.append(len(data))

    # last row (lambda sum constraint)
    for pt_index in range(len(pts)):
        data.append(1)
        inds.append(dims + pt_index)

    indptr.append(len(data))

    csr = csr_matrix((data, inds, indptr), shape=(dims + 1, dims + len(pts)), dtype=float)
    csr.check_format()

    lpi.set_constraints_csr(csr)

    return lpi
