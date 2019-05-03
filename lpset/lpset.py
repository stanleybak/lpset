'''
Lpset implementation

Stanley Bak
'''

from scipy.sparse import csr_matrix
import numpy as np

import matplotlib.pyplot as plt

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

    def __init__(self, csr, rhs, a_mat=None):

        if not isinstance(csr, csr_matrix):
            csr = csr_matrix(csr, dtype=float)

        if not isinstance(rhs, np.ndarray):
            rhs = np.array(rhs, dtype=float)

        self.csr = csr
        self.rhs = rhs

        if a_mat is not None:
            if not isinstance(rhs, np.ndarray):
                a_mat = np.array(a_mat, dtype=float)

            assert csr.shape[1] == a_mat.shape[1], f"csr.shape = {csr.shape}, a_mat.shape = {a_mat.shape}"

            self.a_mat = a_mat
        else:
            self.a_mat = np.identity(self.csr.shape[1], dtype=float)

        self.lpi = self.make_lpi()

        self.freeze_attrs()

    def __str__(self):
        return str(self.lpi)

    def clone(self):
        'clone this lpset'

        return LpSet(self.csr.copy(), self.rhs.copy(), self.a_mat.copy())

    def make_lpi(self):
        'make the lpinstance from csr and rhs'

        dims = self.a_mat.shape[0] # each row in a is a current-time variable

        lpi = LpInstance()
        lpi.add_rows_equal_zero(dims)

        names = [f"i{var_index}" for var_index in range(self.csr.shape[1])]
        names += [f"c{var_index}" for var_index in range(dims)]

        lpi.add_cols(names)

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
        csr = csr_matrix((data, inds, indptr), shape=(dims, num_cols), dtype=float)
        csr.check_format()

        lpi.set_constraints_csr(csr)

        # add constraints on initial conditions, offset by <dims> rows
        lpi.set_constraints_csr(self.csr, offset=(dims, 0))

        cur_vars_offset = self.csr.shape[1]
        lpi.set_reach_vars(dims, (0, 0), cur_vars_offset)

        return lpi

    def verts(self, xdim=0, ydim=1):
        '''get (an approximation of) the vertices of a projection of this lpset

        returns a list of 2-d points that wraps, so that rv[0] == rv[-1]
        '''

        return lpplot.get_verts(self.lpi, xdim=xdim, ydim=ydim, plot_vecs=LpSet.plot_vecs)

    def verts3d(self, xdim=0, ydim=1, zdim=2):
        '''get (an approximation of) the vertices of a 3d projection of this lpset
    
        this returns a list of 3d points
        '''

        return lpplot.get_verts3d(self.lpi, xdim=xdim, ydim=ydim, zdim=zdim)

    def plot(self, color='k-', xdim=0, ydim=1):
        'use matplotlib to plot this lpset'

        verts = self.verts(xdim=xdim, ydim=ydim)

        xs, ys = zip(*verts)
        plt.plot(xs, ys, color)

    def plot3d(self, ax, markercol='ko', edgecol='k-', xdim=0, ydim=1, zdim=2):
        "plot a 3d projection of the zonotope onto the given matplotlib axis object (initialized with proj='3d')"

        pts = self.verts3d(xdim=xdim, ydim=ydim, zdim=zdim)

        ax.plot(pts.T[0], pts.T[1], pts.T[2], markercol, ms=0.5)

        lpplot3d.plot_hull(ax, pts)
        
    def minkowski_difference(other):
        'perform an in-place minkowski difference'

        # algorithm: for every normal direction, optimize that direction in other and move edges inward accordingly
        pass
        


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

def from_zonotope(center, generator_list):
    'make a new lp instance from the passed in zonotope'

    # use fourier-morwitz elmination to get rid of all the generator variables

    # generator constraints
    # -1 + c[0] <= alpha1 <= 1 + c[0]
    # -1 + c[1] <= alpha2 <= 1 + c[1]
    # ...

    # definition constraints
    # x0 = c + alpha1 * g1[0] + alpha2 * g2[0] + ...
    # x1 = c + alpha1 * g1[1] + alpha2 * g2[1] + ...
    # ...

    # we should be able to solve each of the definition constraints for alpha1, then alpha2, etc.

    # first set of variables are alphas, followed by the real variables
    ng = len(generator_list)
    nc = len(center)
    num_vars = ng + nc

    gen_constraints = [[1.0 if col == row else 0.0 for col in range(num_vars)] for row in range(ng)] 



    

    #variables [x1, x2, ..., center_var, alpha1, alpha2, ...]
    mat = []
    rhs = []

    cdims = len(center)
    dims = cdims + len(generator_list)

    # bounds on center_var (center_var = 1)
    #mat.append([1 if d == cdims else 0 for d in range(dims)])
    #rhs.append(1)

    #mat.append([-1 if d == cdims else 0 for d in range(dims)])
    #rhs.append(-1)

    for i, g in enumerate(generator_list):
        assert len(g) == cdims, "expected each generator to have the same number of dims as the center"
        
        # bounds on each alpha (-1 <= alpha <= 1)
        mat.append([1 if d == cdims + i else 0 for d in range(dims)])
        rhs.append(1)

        mat.append([-1 if d == cdims + i else 0 for d in range(dims)])
        rhs.append(1)

    # zonotope generator constraints x = c + alpha1 * g1[0] + alpha2 * g2[0] + ...
    for dim, c in enumerate(center):
        row = [0] * dims

        row[dim] = -1
        #row[cdims] = c

        for gindex, generator in enumerate(generator_list):     
            row[cdims + gindex] = generator[dim]

        mat.append(row)
        rhs.append(-c)

        mat.append([-1 * x for x in row])
        rhs.append(c)

    # project onto the first cdim variables (current time variables)

    # identity matrix with extra columns of zeros
    a_mat = np.zeros((cdims, dims), dtype=float)
    
    for d in range(cdims):
        a_mat[d, d] = 1.0

    return LpSet(mat, rhs, a_mat=a_mat)
