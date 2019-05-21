'''
Lpset implementation

Stanley Bak
'''

from scipy.sparse import csr_matrix, csc_matrix
import scipy as sp
import numpy as np

import matplotlib.pyplot as plt

import swiglpk as glpk

from lpinstance import LpInstance
import lpplot
import lpplot3d
from util import Freezable

class LpSet(Freezable):
    '''lp set container class

    lp set is a linear transformation of a set of linear constraints

    mat is the constraints matirx, rhs is the constraints rhs, a_mat is the linear transformation
    '''

    # override if more or less accuracy is desired for plotting
    plot_vecs = lpplot.make_plot_vecs(num_angles=1024, offset=0.1)

    def __init__(self, csr, rhs, a_mat=None, types=None, names=None, dims=None):

        if not isinstance(csr, csr_matrix):
            csr = csr_matrix(csr, dtype=float)

        if not isinstance(rhs, np.ndarray):
            rhs = np.array(rhs, dtype=float)

        self.csr = csr
        self.rhs = rhs

        if dims is None:
            dims = self.csr.shape[1]

        if a_mat is not None:
            if not isinstance(rhs, np.ndarray):
                a_mat = np.array(a_mat, dtype=float)

            assert dims == a_mat.shape[1], f"dims = {dims}, a_mat.shape = {a_mat.shape}"

            self.a_mat = a_mat
        else: 
            self.a_mat = np.identity(dims, dtype=float)

        self.lpi = self.make_lpi(types=types, names=names)

        self.freeze_attrs()

    def __str__(self):
        return str(self.lpi)

    def clone(self):
        'clone this lpset'

        return LpSet(self.csr.copy(), self.rhs.copy(), self.a_mat.copy())

    def make_lpi(self, types=None, names=None):
        'make the lpinstance from csr and rhs'

        dims = self.a_mat.shape[0] # each row in a is a current-time variable

        lpi = LpInstance()
        lpi.add_rows_equal_zero(dims)
        
        if not names:
            names = [f"i{var_index}" for var_index in range(self.csr.shape[1])]
        else:
            assert len(names) == self.csr.shape[1]

        names += [f"c{var_index}" for var_index in range(dims)]

        lpi.add_cols(names)

        if types:
            lpi.add_rows_with_types(types, self.rhs)
        else:
            lpi.add_rows_less_equal(self.rhs)

        # make constraints as csr_matrix
        data = []
        inds = []
        indptr = [0]

        # BM 0 -I for first n rows
        for row_index, row in enumerate(self.a_mat):
            for col_index, val in enumerate(row):
                if val != 0:
                    data.append(val)
                    inds.append(col_index)

            data.append(-1)
            inds.append(self.csr.shape[1] + row_index)

            indptr.append(len(data))

        num_cols = self.csr.shape[1] + dims
        csr_bm = csr_matrix((data, inds, indptr), shape=(dims, num_cols), dtype=float)
        csr_bm.check_format()

        lpi.set_constraints_csr(csr_bm)

        # add constraints on initial conditions, offset by <dims> rows
        lpi.set_constraints_csr(self.csr, offset=(dims, 0))

        cur_vars_offset = self.csr.shape[1]
        basis_mat_rect = (0, 0, self.csr.shape[1], dims)
        lpi.set_reach_vars(dims, basis_mat_rect, cur_vars_offset)

        return lpi

    def get_complement_lpsets(self):
        '''
        get a list of lpi sets, such that their union is the complement of this set
        '''

        cur_dims = self.a_mat.shape[0]
        rank = np.linalg.matrix_rank(self.a_mat)

        # a_mat must be full rank for this algorimth to work (otherwise we could extend A, not implemented)
        assert rank == cur_dims, f"Rank={rank} was not cur_dims={cur_dims}"

        lpsets = []

        for i in range(self.csr.shape[0]):
            # add constraints 0 to i-1 and negate constraint i
                    
            new_csr = sp.sparse.vstack([self.csr[:i, :], -self.csr[i:i+1, :]])
            new_rhs = np.hstack([self.rhs[:i], -self.rhs[i:i+1]])

            lpsets.append(LpSet(new_csr, new_rhs, self.a_mat))

        return lpsets

    def find_point_in_diff(self, b_lpset):
        '''find a point the minkowski difference, self - b_set

        this can fail. currently it just tries the origin
        '''

        c_list = self.get_complement_lpsets()

        # check if b_lpset doesn't intersect any of the lpsets in c
        origin_ok = True
        
        for i, c in enumerate(c_list):
            lpi = intersection(c, b_lpset)

            pt = lpi.minimize(fail_on_unsat=False)

            if pt is not None:
                print(f"complement lpi #{i} is feasible at pt {pt}:\n\n{lpi}")
                origin_ok = False
                break

        assert origin_ok, "the origin is not in the minkowski difference (more clever strategy unimplemented)"

        return np.zeros((self.a_mat.shape[0],), dtype=float)
            
    def verts(self, xdim=0, ydim=1):
        '''get (an approximation of) the vertices of a projection of this lpset

        returns a list of 2-d points that wraps, so that rv[0] == rv[-1]
        '''

        #return lpplot.get_verts(self.lpi, xdim=xdim, ydim=ydim, plot_vecs=LpSet.plot_vecs)
        return lpplot.get_verts_nd(self.lpi, 2)

    def verts3d(self, xdim=0, ydim=1, zdim=2):
        '''get (an approximation of) the vertices of a 3d projection of this lpset
    
        this returns a list of 3d points
        '''

        return lpplot.get_verts_nd(self.lpi, 3)

    def plot(self, color='k-', xdim=0, ydim=1):
        'use matplotlib to plot this lpset'

        print(".todo use new verts() in plot()")
        verts = lpplot.get_verts(self.lpi, xdim=xdim, ydim=ydim, plot_vecs=LpSet.plot_vecs)
                
#        verts = self.verts(xdim=xdim, ydim=ydim)

        xs, ys = zip(*verts)
        plt.plot(xs, ys, color)

    def plot3d(self, ax, xdim=0, ydim=1, zdim=2):
        "plot a 3d projection of the zonotope onto the given matplotlib axis object (initialized with proj='3d')"

        pts = self.verts3d(xdim=xdim, ydim=ydim, zdim=zdim)

        ax.plot(pts.T[0], pts.T[1], pts.T[2], 'k', ms=0.7)

        lpplot3d.plot_hull(ax, pts)
        
    def minkowski_difference(self, other):
        'perform an in-place minkowski difference'

        # algorithm: for every normal direction, optimize that direction in other and move edges inward accordingly

        for row in self.csr:
            row_vec = row.toarray()

            # figure out
        
def from_box(dim_interval_list):
    'make a new lpset from a passed-in box'

    rhs = []

    for lb, ub in dim_interval_list:
        assert lb <= ub, "lower bound ({}) > upper bound ({})".format(lb, ub)
        rhs.append(-lb)
        rhs.append(ub)

    # make constraints as csr_matrix
    dims = len(dim_interval_list)
    data = []
    inds = []
    indptr = [0]

    # -1 <= -lb
    # 1 <= ub
    for n in range(dims):
        data.append(-1)
        inds.append(n)
        indptr.append(len(data))

        data.append(1)
        inds.append(n)
        indptr.append(len(data))

    csr = csr_matrix((data, inds, indptr), shape=(2*dims, dims), dtype=float)
    csr.check_format()

    return LpSet(csr, rhs)

def from_centered_zonotope(generator_list):
    'make a new lp instance from a zontope center and list of generator'

    # generator constraints
    # -1 <= alpha1 <= 1
    # -1 <= alpha2 <= 1
    # ...

    # a_mat is equal to tranpose of the matrix from the generators
    
    # first set of variables are alphas, followed by the real variables
    num_generators = len(generator_list)
    cdims = len(generator_list[0])

    mat = []
    rhs = []

    for i, g in enumerate(generator_list):
        assert len(g) == cdims, "expected each generator to have the same number of dims"
        
        # bounds on each alpha (-1 <= alpha <= 1)
        mat.append([1 if d == i else 0 for d in range(num_generators)])
        rhs.append(1)

        mat.append([-1 if d == i else 0 for d in range(num_generators)])
        rhs.append(1)

    a_mat = np.array(generator_list, dtype=float).transpose()

    assert a_mat.shape == (cdims, num_generators)

    return LpSet(mat, rhs, a_mat=a_mat)

def contains_point(lpset, pt):
    '''does the passed-in lpset contain the passed-in point? This will copy the lpinstance, so it's relatively slow'''

    # intersect with the given point
    # clones and adds constaints x == pt[0], y == pt[1], etc, then check if feasible

    lpi = lpset.lpi.clone()

    assert len(pt) == lpi.dims
    lpi.add_rows_equal(pt)

    # set constraints
    mat = sp.sparse.identity(lpi.num_dims, dtype=float, format='csr')

    lpi.set_constraints_csr(mat, offset=lpi.cur_vars_offset)

    return lpi.is_feasible()

def from_chull(set_a, set_b):
    'construct an lpset from a convex hull of two sets'

    lpi_list = [set_a.lpi, set_b.lpi]

    # base case: exactly two lpis in lpi_list
    assert len(lpi_list) == 2

    dims = lpi_list[0].dims
    assert lpi_list[1].dims == dims

    lpi1 = lpi_list[0]
    csr1 = lpi1.get_full_constraints()
    rhs1 = lpi1.get_rhs()
    types1 = lpi1.get_types()

    # csr1 contains L_left A L_right, split it into these three in order to construct L
    l_left = csr1[:, 0:lpi1.cur_vars_offset]
    a1 = csr1[:, lpi1.cur_vars_offset:lpi1.cur_vars_offset+dims]
    l_right = csr1[:, lpi1.cur_vars_offset+dims:]

    l1 = sp.sparse.hstack([l_left, l_right])

    # repeat for lpi2
    lpi2 = lpi_list[1]
    csr2 = lpi2.get_full_constraints()
    rhs2 = lpi2.get_rhs()
    types2 = lpi2.get_types()

    # csr1 contains L_left A L_right, split it into these three in order to construct L
    l_left = csr2[:, 0:lpi2.cur_vars_offset]
    a2 = csr2[:, lpi2.cur_vars_offset:lpi2.cur_vars_offset+dims]
    l_right = csr2[:, lpi2.cur_vars_offset+dims:]
    l2 = sp.sparse.hstack([l_left, l_right])

    lpi = LpInstance()
    #lpi.add_rows_equal_zero(dims)

    types = types1 + types2
    rhs = [n for n in rhs1] + ([0] * len(rhs2))

    lpi.add_rows_with_types(types, rhs)

    cols = []

    cols += [f"A1_{i}" for i in range(dims)]
    cols += [f"A2_{i}" for i in range(dims)]
    cols += [f"L1_{i}" for i in range(l1.shape[1])]
    cols += [f"L2_{i}" for i in range(l2.shape[1])]
    cols += ["a"]

    lpi.add_cols(cols)

    # set constraints
    l2_zero = csr_matrix((a1.shape[0], l2.shape[1])) # the 0 above L2
    l1_zero = csr_matrix((a2.shape[0], l1.shape[1])) # the 0 below L1
    a1_zero = csr_matrix((a2.shape[0], a1.shape[1])) # the 0 below A1

    rhs1_vmat = csr_matrix(np.array([[num] for num in rhs1]))
    rhs2_vmat = csr_matrix(np.array([[num] for num in rhs2]))

    top = csr_matrix(sp.sparse.hstack([a1, a1, l1, l2_zero, rhs1_vmat]))
    lpi.set_constraints_csr(top)

    bottom = csr_matrix(sp.sparse.hstack([a1_zero, -a2, l1_zero, l2, -rhs2_vmat]))
    lpi.set_constraints_csr(bottom, offset=(top.shape[0], 0))

    lpi.dims = dims
    lpi.cur_vars_offset = 0 # left-most variables are current time variables

    # need to essentially take snapshot variables
    csr = lpi.get_full_constraints()
    rhs = lpi.get_rhs()
    types = lpi.get_types()

    # from_constraints assumes left-most variables are current-time variables
    return LpSet(csr, rhs, types=types, names=cols, dims=lpi.dims)

def from_forced_chull(set_a, set_b, lam=None, slack=0.0):
    'construct an lpset from a convex hull of two sets, where the lambda is forced to some value'

    lpi_list = [set_a.lpi, set_b.lpi]

    # base case: exactly two lpis in lpi_list
    assert len(lpi_list) == 2

    dims = lpi_list[0].dims
    assert lpi_list[1].dims == dims

    lpi1 = lpi_list[0]
    csr1 = lpi1.get_full_constraints()
    rhs1 = lpi1.get_rhs()
    types1 = lpi1.get_types()

    # csr1 contains L_left A L_right, split it into these three in order to construct L
    l_left = csr1[:, 0:lpi1.cur_vars_offset]
    a1 = csr1[:, lpi1.cur_vars_offset:lpi1.cur_vars_offset+dims]
    l_right = csr1[:, lpi1.cur_vars_offset+dims:]

    l1 = sp.sparse.hstack([l_left, l_right])

    # repeat for lpi2
    lpi2 = lpi_list[1]
    csr2 = lpi2.get_full_constraints()
    rhs2 = lpi2.get_rhs()
    types2 = lpi2.get_types()

    # csr1 contains L_left A L_right, split it into these three in order to construct L
    l_left = csr2[:, 0:lpi2.cur_vars_offset]
    a2 = csr2[:, lpi2.cur_vars_offset:lpi2.cur_vars_offset+dims]
    l_right = csr2[:, lpi2.cur_vars_offset+dims:]
    l2 = sp.sparse.hstack([l_left, l_right])

    lpi = LpInstance()
    #lpi.add_rows_equal_zero(dims)

    types = types1 + types2

    # old:
    rhs = [n for n in rhs1] + ([0] * len(rhs2))
    #rhs = [0.3 * n for n in rhs1] + ([0] * len(rhs2))

    lpi.add_rows_with_types(types, rhs)

    cols = []

    cols += [f"A1_{i}" for i in range(dims)]
    cols += [f"A2_{i}" for i in range(dims)]
    cols += [f"L1_{i}" for i in range(l1.shape[1])]
    cols += [f"L2_{i}" for i in range(l2.shape[1])]
    cols += ["a"]

    lpi.add_cols(cols)

    # set constraints
    l2_zero = csr_matrix((a1.shape[0], l2.shape[1])) # the 0 above L2
    l1_zero = csr_matrix((a2.shape[0], l1.shape[1])) # the 0 below L1
    a1_zero = csr_matrix((a2.shape[0], a1.shape[1])) # the 0 below A1

    rhs1_vmat = csr_matrix(np.array([[num] for num in rhs1]))
    rhs2_vmat = csr_matrix(np.array([[num] for num in rhs2]))

    top = csr_matrix(sp.sparse.hstack([a1, a1, l1, l2_zero, rhs1_vmat]))
    lpi.set_constraints_csr(top)

    bottom = csr_matrix(sp.sparse.hstack([a1_zero, -a2, l1_zero, l2, -rhs2_vmat]))
    lpi.set_constraints_csr(bottom, offset=(top.shape[0], 0))

    # add slack variables
    nrows = lpi.get_num_rows()
    ncols = lpi.get_num_cols()

    new_col_names = [f"s{i}" for i in range(nrows)]
    lpi.add_cols(new_col_names)

    cols += new_col_names

    # add rows for slack variable bounds
    slack_rhs = []

    for _ in range(nrows):
        slack_rhs.append(slack)
        slack_rhs.append(0)

    lpi.add_rows_less_equal(slack_rhs)

    # add constraints on slack variables
    data = []
    indices = []
    indptr = [0]

    for sindex in range(nrows):
        indices.append(sindex)
        data.append(-1.0)

        indices.append(nrows + 2*sindex)
        data.append(1.0)

        indices.append(nrows + 2*sindex + 1)
        data.append(-1.0)

        indptr.append(len(data))

    slack_csc = csc_matrix((data, indices, indptr), shape=(3*nrows, nrows), dtype=float)
    slack_csc.check_format()
    
    lpi.set_constraints_csc(slack_csc, offset=(0, ncols))

    # add lamda constraint
    if lam is not None:
        nrows = lpi.get_num_rows()
        lpi.add_rows_equal([lam])
        l = [0] * top.shape[1]
        l[-1] = 1.0
        mat = csr_matrix(np.array(l, dtype=float))
        lpi.set_constraints_csr(mat, offset=(nrows, 0))

    # need to essentially take snapshot variables
    csr = lpi.get_full_constraints()
    rhs = lpi.get_rhs()
    types = lpi.get_types()

    # from_constraints assumes left-most variables are current-time variables
    return LpSet(csr, rhs, types=types, names=cols, dims=set_a.lpi.dims)

def intersection(a, b):
    '''return an lpi that is the minkowski sum of the two passed-in lpsets'''

    lpi = a.lpi.clone()
    lpi.basis_mat_rect = [0, 0, 0, 0] # no basis mat, since it's not a proper lpset

    nrows = lpi.get_num_rows()
    ncols = lpi.get_num_cols()

    dims = b.a_mat.shape[0]
    
    lpi.add_rows_equal_zero(dims)

    names = [f"b_i{x}" for x in range(b.a_mat.shape[1])]
    lpi.add_cols(names)

    lpi.add_rows_less_equal(b.rhs)

    # make constraints as csr_matrix
    data = []
    inds = []
    indptr = [0]

    # 0 -I BM for n rows
    for row_index, row in enumerate(b.a_mat):

        data.append(-1)
        inds.append(ncols - dims + row_index)
        
        for col_index, val in enumerate(row):
            if val != 0:
                data.append(val)
                inds.append(ncols + col_index)

        indptr.append(len(data))

    num_cols = ncols + b.a_mat.shape[1]
    num_rows = dims
    csr_bm = csr_matrix((data, inds, indptr), shape=(num_rows, num_cols), dtype=float)
    csr_bm.check_format()

    lpi.set_constraints_csr(csr_bm, offset=(nrows, 0))

    # add constraints on initial conditions, offset by <dims> rows
    lpi.set_constraints_csr(b.csr, offset=(nrows + dims, ncols))

    return lpi

def minkowski_sum(lpset_list):
    '''
    perform a minkowski sum of the passed-in sets, and return the resultant lpi
    '''

    lpi_list = [s.lpi for s in lpset_list]

    for lpi in lpi_list:
        assert lpi.dims == lpi_list[0].dims, "dimension mismatch during minkowski sum"

    dims = lpi_list[0].dims

    csr_list = []
    combined_rhs = [0] * dims
    combined_types = [glpk.GLP_FX] * dims
    combined_names = [f"c{n}" for n in range(dims)]

    total_new_vars = dims

    for i, lpi in enumerate(lpi_list):
        csr = lpi.get_full_constraints()
        csr_list.append(csr)
        combined_rhs += [v for v in lpi.get_rhs()]
        combined_types += lpi.get_types()

        total_new_vars += csr.shape[1]
        combined_names += [f"l{i}_{v}" for v in range(csr.shape[1])]

    # create combined_csr constraints
    data = []
    indices = []
    indptr = [0]

    for d in range(dims):
        data.append(1)
        indices.append(d)
        col_offset = dims

        for lpi in lpi_list:
            data.append(-1)
            indices.append(col_offset + lpi.cur_vars_offset + d)

            col_offset += lpi.get_num_cols()

        indptr.append(len(data))

    # copy constraints from each lpi
    col_offset = dims
    indptr_offset = indptr[-1]
    
    for csr in csr_list:
        data += [d for d in csr.data]
        indices += [col_offset + i for i in csr.indices]
        indptr += [indptr_offset + i for i in csr.indptr[1:]]

        col_offset += csr.shape[1]
        indptr_offset = indptr[-1]

    rows = len(combined_rhs)
    cols = col_offset
    combined_csr = csr_matrix((data, indices, indptr), shape=(rows, cols), dtype=float)

    return LpSet(combined_csr, combined_rhs, types=combined_types, names=combined_names, dims=dims)
