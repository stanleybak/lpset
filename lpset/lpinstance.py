'''
Stanley Bak
May 2018
GLPK python interface using swiglpk
'''

from scipy.sparse import csr_matrix, csc_matrix
import numpy as np

from termcolor import colored

import swiglpk as glpk
from util import Freezable

class LpInstance(Freezable):
    'Linear programming wrapper using glpk (through swiglpk python interface)'

    def __init__(self):
        'initialize the lp instance'

        self.lp = glpk.glp_create_prob() # pylint: disable=invalid-name

        # these are assigned on set_reach_vars()
        self.dims = None
        self.basis_mat_rect = None # 4-tuple, x, y, w, h
        self.cur_vars_offset = None

        # internal bookkeeping
        self.obj_cols = [] # columns in the LP with an assigned objective coefficient
        self.names = [] # column names

        self.freeze_attrs()

    def __del__(self):
        if hasattr(self, 'lp') and self.lp is not None:
            glpk.glp_delete_prob(self.lp)
            self.lp = None

    def clone(self):
        'create a copy of this lp instance'

        rv = LpInstance()
        glpk.glp_copy_prob(rv.lp, self.lp, glpk.GLP_ON)

        # copy internal bookkeeping
        rv.obj_cols = self.obj_cols.copy()
        rv.names = self.names.copy()
        
        rv.set_reach_vars(self.dims, self.basis_mat_rect, self.cur_vars_offset)

        return rv

    def set_reach_vars(self, dims, basis_mat_rect, cur_vars_offset):
        'set reachability variables'

        num_rows = self.get_num_rows()
        num_cols = self.get_num_cols()

        assert len(basis_mat_rect) == 4
        assert dims == basis_mat_rect[3]
        assert basis_mat_rect[1] + dims <= num_rows
        assert basis_mat_rect[0] + 2 * dims <= num_cols  # need >= 2*dims for cur_time vars somewhere to the right of BM

        self.dims = dims
        self.basis_mat_rect = basis_mat_rect
        self.cur_vars_offset = cur_vars_offset #num_cols - dims # right-most variables

    def _column_names_str(self, cur_var_print):
        'get the line in __str__ for the column names'

        rv = "   "

        for col, name in enumerate(self.names):
            name = self.names[col]
            name = "-" if name is None else name
            
            if len(name) < 6:
                name = (" " * (6 - len(name))) + name
            else:
                name = name[0:6]

            if self.cur_vars_offset <= col < self.cur_vars_offset + self.dims: 
                rv += cur_var_print(name) + " "
            else:
                rv += name + " "

        rv += "\n"
        
        return rv

    def _opt_dir_str(self, zero_print):
        'get the optimization direction line for __str__'

        lp = self.lp
        cols = self.get_num_cols()
        rv = "min"

        for col in range(1, cols + 1):
            val = glpk.glp_get_obj_coef(lp, col)
            num = str(val)
            
            if len(num) < 6:
                num = (" " * (6 - len(num))) + num
            else:
                num = num[0:6]

            if val == 0:
                rv += zero_print(num) + " "
            else:
                rv += num + " "

        rv += "\n"
        
        return rv

    def _col_stat_str(self):
        'get the column statuses line for __str__'

        lp = self.lp
        cols = self.get_num_cols()

        stat_labels = ["?(0)?", "BS", "NL", "NU", "NF", "NS", "?(6)?"]
        rv = "   "

        for col in range(1, cols + 1):
            rv += "{:>6} ".format(stat_labels[glpk.glp_get_col_stat(lp, col)])

        rv += "\n"

        return rv

    def _constraints_str(self, bm_print, input_print, zero_print):
        'get the constraints matrix lines for __str__'

        rv = ""
        lp = self.lp
        rows = self.get_num_rows()
        cols = self.get_num_cols()
        
        stat_labels = ["?(0)?", "BS", "NL", "NU", "NF", "NS"]
        inds = glpk.intArray(cols + 1)
        vals = glpk.doubleArray(cols + 1)

        for row in range(1, rows + 1):
            stat = glpk.glp_get_row_stat(lp, row)
            assert 0 <= stat <= len(stat_labels)
            rv += "{:2}: {} ".format(row, stat_labels[stat])

            num_inds = glpk.glp_get_mat_row(lp, row, inds, vals)

            for col in range(1, cols + 1):
                val = 0

                for index in range(1, num_inds+1):
                    if inds[index] == col:
                        val = vals[index]
                        break

                num = str(val)
                if len(num) < 6:
                    num = (" " * (6 - len(num))) + num
                else:
                    num = num[0:6]

                if self.basis_mat_rect[1] <= row - 1 < self.basis_mat_rect[0] + self.basis_mat_rect[3] and \
                        self.basis_mat_rect[0] <= col - 1 < self.basis_mat_rect[1] + self.basis_mat_rect[2]:
                    rv += bm_print(num) + " "
                else:
                    rv += (zero_print(num) if val == 0 else num) + " "

            row_type = glpk.glp_get_row_type(lp, row)

            if row_type == glpk.GLP_FX:
                val = glpk.glp_get_row_ub(lp, row)
                rv += " == "
            else:
                assert row_type == glpk.GLP_UP
                
                val = glpk.glp_get_row_ub(lp, row)
                rv += " <= "

            num = str(val)
            if len(num) < 6:
                num = (" " * (6 - len(num))) + num
            else:
                num = num[0:6]

            rv += (zero_print(num) if val == 0 else num) + " "

            rv += "\n"

        return rv

    def __str__(self, plain_text=False):
        'get the LP as string (useful for debugging)'

        if plain_text:
            cur_var_print = bm_print = input_print = zero_print = lambda x: x
        else:
            def cur_var_print(s):
                'print function for current variables'

                return colored(s, on_color="on_cyan")

            def bm_print(s):
                'print function for basis matrix'

                return colored(s, on_color="on_red")

            def input_print(s):
                'print function for input offset'

                return colored(s, on_color="on_green")

            def zero_print(s):
                'print function for zeros'

                return colored(s, 'white', attrs=['dark'])

        rows = self.get_num_rows()
        cols = self.get_num_cols()
        rv = "Lp has {} columns (variables) and {} rows (constraints)\n".format(cols, rows)

        rv += self._column_names_str(cur_var_print)

        rv += self._opt_dir_str(zero_print)

        rv += "subject to:\n"

        rv += self._col_stat_str()

        rv += self._constraints_str(bm_print, input_print, zero_print)
        
        rv += "Key: " + bm_print("Basis Matrix") + " " + cur_var_print("Cur Vars") + "\n"

        return rv

    def get_num_rows(self):
        'get the number of rows in the lp'

        return glpk.glp_get_num_rows(self.lp)

    def get_num_cols(self):
        'get the number of columns in the lp'

        #return glpk.glp_get_num_cols(self.lp)
        return len(self.names) # probably faster than making a call to glpk

    def add_rows_equal(self, rhs_vec):
        '''add rows to the LP with == rhs[i] constraints'''

        num = len(rhs_vec)

        if num > 0:
            num_rows = glpk.glp_get_num_rows(self.lp)

            # create new row for each constraint
            glpk.glp_add_rows(self.lp, num)

            for i, val in enumerate(rhs_vec):
                glpk.glp_set_row_bnds(self.lp, num_rows + i + 1, glpk.GLP_FX, val, val)  # '== val' constraints

    def add_rows_equal_zero(self, num):
        '''add rows to the LP with == 0 constraints'''

        self.add_rows_equal(num * [0])

    def add_rows_less_equal(self, rhs_vec):
        '''add rows to the LP with <= constraints

        rhs_vector is the right-hand-side values of the constriants
        '''

        if isinstance(rhs_vec, list):
            rhs_vec = np.array(rhs_vec, dtype=float)

        assert isinstance(rhs_vec, np.ndarray) and len(rhs_vec.shape) == 1, "expected 1-d right-hand-side vector"

        if rhs_vec.shape[0] > 0:
            num_rows = glpk.glp_get_num_rows(self.lp)

            # create new row for each constraint
            glpk.glp_add_rows(self.lp, len(rhs_vec))

            for i, rhs in enumerate(rhs_vec):
                glpk.glp_set_row_bnds(self.lp, num_rows + i + 1, glpk.GLP_UP, 0, rhs)  # '<=' constraint

    def get_types(self):
        '''get the constraint types. These are swiglpk.GLP_FX, swiglpk.GLP_UP, or swiglpk.GLP_LO'''

        lp_rows = glpk.glp_get_num_rows(self.lp)
        rv = []

        for row in range(lp_rows):
            rv.append(glpk.glp_get_row_type(self.lp, row + 1))

        return rv

    def add_rows_with_types(self, types, rhs_vec):
        '''add rows to the LP with the given types

        types is a vector of types: swiglpk.GLP_FX, swiglpk.GLP_UP, or swiglpk.GLP_LO
        rhs_vector is the right-hand-side values of the constriants
        '''

        assert len(types) == len(rhs_vec)

        if isinstance(rhs_vec, list):
            rhs_vec = np.array(rhs_vec, dtype=float)

        assert isinstance(rhs_vec, np.ndarray) and len(rhs_vec.shape) == 1, "expected 1-d right-hand-side vector"

        if rhs_vec.shape[0] > 0:
            num_rows = glpk.glp_get_num_rows(self.lp)

            # create new row for each constraint
            glpk.glp_add_rows(self.lp, len(rhs_vec))
            
            for i, pair in enumerate(zip(rhs_vec, types)):
                rhs, ty = pair

                if ty == glpk.GLP_UP:
                    glpk.glp_set_row_bnds(self.lp, num_rows + i + 1, glpk.GLP_UP, 0, rhs)  # '<=' constraint
                elif ty == glpk.GLP_LO:
                    glpk.glp_set_row_bnds(self.lp, num_rows + i + 1, glpk.GLP_LO, rhs, 0)  # '>=' constraint
                else:
                    assert ty == glpk.GLP_FX, f"got unknown row type: {ty}"

                    glpk.glp_set_row_bnds(self.lp, num_rows + i + 1, glpk.GLP_FX, rhs, rhs)  # '>=' constraint

    def add_cols(self, names):
        'add a certain number of columns to the LP'

        assert isinstance(names, list)
        num_vars = len(names)

        if num_vars > 0:
            num_cols = self.get_num_cols()

            self.names += names
            glpk.glp_add_cols(self.lp, num_vars)

            for i in range(num_vars):
                glpk.glp_set_col_bnds(self.lp, num_cols + i + 1, glpk.GLP_FR, 0, 0)  # free variable (-inf, inf)

    def set_constraints_csc(self, csc_mat, offset=None):
        '''set the constrains column by column to be equal to the passed-in csc matrix

        offset is an optional tuple (num_rows, num_cols) which tells you the top-left offset for the assignment
        '''

        assert isinstance(csc_mat, csc_matrix)
        assert csc_mat.dtype == float

        if offset is None:
            offset = (0, 0)

        assert len(offset) == 2, "offset should be a 2-tuple (num_rows, num_cols)"

        # check that the matrix is in bounds
        lp_rows = self.get_num_rows()
        lp_cols = self.get_num_cols()

        if offset[0] < 0 or offset[1] < 0 or \
                            offset[0] + csc_mat.shape[0] > lp_rows or offset[1] + csc_mat.shape[1] > lp_cols:
            raise RuntimeError(("Error: set constraints matrix out of bounds (offset was " + \
                "{}, matrix size was {}), but lp size was ({}, {})").format(
                    offset, csc_mat.shape, lp_rows, lp_cols))

        # actually set the constraints col by col
        indptr = csc_mat.indptr
        indices = csc_mat.indices
        data_list = csc_mat.data.tolist()

        for col in range(csc_mat.shape[1]):
            # we must copy the indices since glpk is offset by 1 :(
            count = int(indptr[col + 1] - indptr[col])

            indices_list = [1 + offset[0] + int(indices[index]) for index in range(indptr[col], indptr[col+1])]
            indices_vec = SwigArray.as_int_array(indices_list)

            data_row_list = data_list[indptr[col]:indptr[col+1]]
            data_vec = SwigArray.as_double_array(data_row_list)

            glpk.glp_set_mat_col(self.lp, offset[1] + col + 1, count, indices_vec, data_vec)

    def set_constraints_csr(self, csr_mat, offset=None):
        '''set the constrains row by row to be equal to the passed-in csr matrix

        offset is an optional tuple (num_rows, num_cols) which tells you the top-left offset for the assignment
        '''

        assert isinstance(csr_mat, csr_matrix)
        assert csr_mat.dtype == float

        if offset is None:
            offset = (0, 0)

        assert len(offset) == 2, "offset should be a 2-tuple (num_rows, num_cols)"

        # check that the matrix is in bounds
        lp_rows = self.get_num_rows()
        lp_cols = self.get_num_cols()

        if offset[0] < 0 or offset[1] < 0 or \
                            offset[0] + csr_mat.shape[0] > lp_rows or offset[1] + csr_mat.shape[1] > lp_cols:
            raise RuntimeError("Error: set constraints matrix out of bounds (offset was " + \
                "{}, matrix size was {}), but lp size was ({}, {})".format(
                    offset, csr_mat.shape, lp_rows, lp_cols))

        # actually set the constraints row by row
        indptr = csr_mat.indptr
        indices = csr_mat.indices
        data_list = csr_mat.data.tolist()

        for row in range(csr_mat.shape[0]):
            # we must copy the indices since glpk is offset by 1 :(
            count = int(indptr[row + 1] - indptr[row])

            indices_list = [1 + offset[1] + int(indices[index]) for index in range(indptr[row], indptr[row+1])]
            indices_vec = SwigArray.as_int_array(indices_list)

            data_row_list = data_list[indptr[row]:indptr[row+1]]
            data_vec = SwigArray.as_double_array(data_row_list)

            glpk.glp_set_mat_row(self.lp, offset[0] + row + 1, count, indices_vec, data_vec)

    def get_dense_constraints(self, x, y, w, h):
        'get a subconstraint matrix from the lpi as a dense matrix'

        rv = np.zeros((h, w))

        lp_rows = self.get_num_rows()
        lp_cols = self.get_num_cols()

        assert x >= 0 and w >= 0 and x + w <= lp_cols, "invalid x range requested"
        assert y >= 0 and h >= 0 and y + h <= lp_rows, "invalid y range requested"

        inds_row = glpk.intArray(lp_cols + 1)
        vals_row = glpk.doubleArray(lp_cols + 1)

        for row in range(y + 1, y + h + 1):
            row_offset = row - (y + 1)
            got_len = glpk.glp_get_mat_row(self.lp, row, inds_row, vals_row)

            for i in range(1, got_len+1):
                if inds_row[i] > x and inds_row[i] <= x + w:
                    col_offset = inds_row[i] - (x + 1)
                    rv[row_offset, col_offset] = vals_row[i]
                    
        return rv

    def get_rhs(self, row_indices=None):
        '''get the rhs vector of the constraints

        row_indices - a list of requested indices (None=all)

        this returns an np.array of rhs values for the requested indices
        '''

        rv = []

        if row_indices is None:
            lp_rows = glpk.glp_get_num_rows(self.lp)
            row_indices = range(lp_rows)

        for row in row_indices:
            row_type = glpk.glp_get_row_type(self.lp, row + 1)

            if row_type in [glpk.GLP_FX, glpk.GLP_UP]:
                limit = glpk.glp_get_row_ub(self.lp, row + 1)
            elif row_type == glpk.GLP_LO:
                limit = glpk.glp_get_row_ub(self.lp, row + 1)
            else:
                raise RuntimeError("Error: Unsupported type ({}) in getRhs() in row {}".format(row_type, row))

            rv.append(limit)

        return np.array(rv, dtype=float)

    def get_full_constraints(self):
        '''get the LP matrix as a csr_matrix
        '''

        lp_rows = self.get_num_rows()
        lp_cols = self.get_num_cols()
        nnz = glpk.glp_get_num_nz(self.lp)

        data = np.zeros((nnz,), dtype=float)
        inds = np.zeros((nnz,), dtype=np.int32)
        indptr = np.zeros((lp_rows+1,), dtype=np.int32)

        inds_row = glpk.intArray(lp_cols + 1)
        vals_row = glpk.doubleArray(lp_cols + 1)
        data_index = 0
        indptr[0] = 0

        for row in range(1, lp_rows + 1):
            got_len = glpk.glp_get_mat_row(self.lp, row, inds_row, vals_row)

            for i in range(1, got_len + 1):
                data[data_index] = vals_row[i]
                inds[data_index] = inds_row[i] - 1
                data_index += 1

            indptr[row] = data_index

        csr_mat = csr_matrix((data, inds, indptr), shape=(lp_rows, lp_cols), dtype=float)
        csr_mat.check_format()

        return csr_mat

    def is_feasible(self):
        '''check if the lp is feasible

        returns a feasible point or None
        '''

        return self.minimize(columns=[], fail_on_unsat=False) is not None

    def set_minimize_direction(self, direction_vec, is_csr=False, offset=None):
        '''set the direction for the optimization

        if offset is None, will use cur_vars_offset (direction is in terms of current-time variables)
        '''

        if offset is None:
            offset = self.cur_vars_offset

            size = direction_vec.shape[1] if is_csr else len(direction_vec)

            assert size <= self.dims, "len(direction_vec) ({}) > number of cur_vars({})".format(
                size, self.dims)
        else:
            assert len(direction_vec) + offset <= self.get_num_cols(), \
                  f"col {len(direction_vec) + offset} out of bounds (>= {self.get_num_cols()}"

        # set the previous objective columns to zero
        for i in self.obj_cols:
            glpk.glp_set_obj_coef(self.lp, i, 0)

        self.obj_cols = []

        if is_csr:
            assert isinstance(direction_vec, csr_matrix)
            assert direction_vec.shape[0] == 1

            data, inds, indptr = direction_vec.data, direction_vec.indices, direction_vec.indptr
            
            for n in range(indptr[1]):
                col = int(1 + offset + inds[n])
                self.obj_cols.append(col)

                if col > len(self.names):
                    print(self)
                    
                assert col <= len(self.names) 
                glpk.glp_set_obj_coef(self.lp, col, data[n])

        else: # non-csr
            if not isinstance(direction_vec, np.ndarray):
                direction_vec = np.array(direction_vec, dtype=float)

            assert len(direction_vec.shape) == 1, f"dir_vec shape was {direction_vec.shape}"

            for i, direction in enumerate(direction_vec):
                col = int(1 + offset + i)
                self.obj_cols.append(col)
                glpk.glp_set_obj_coef(self.lp, col, float(direction))

    def minimize(self, direction_vec=None, columns=None, fail_on_unsat=True):
        '''minimize the lp, returning a list of assigments to each of the variables

        if direction_vec is not None, this will first assign the optimization direction (note: relative to cur_vars)
        if columns is not None, will only return the requested columns (default: all columns)
        if fail_on_unsat is True and the LP is infeasible, an UnsatError is raised
        unsat (sometimes happens in GLPK due to likely bug, see space station model)

        returns None if UNSAT, otherwise the optimization result. Use columns=[] if you're not interested in the result
        '''

        if direction_vec is not None:
            self.set_minimize_direction(direction_vec)

        # setup lp params
        params = glpk.glp_smcp()
        glpk.glp_init_smcp(params)
        params.meth = glpk.GLP_DUALP # use dual simplex since we're reoptimizing often
        params.msg_lev = glpk.GLP_MSG_OFF
        params.tm_lim = 1000 # 1000 ms time limit

        simplex_res = glpk.glp_simplex(self.lp, params)

        # process simplex result
        rv = self._process_simplex_result(simplex_res, columns)

        if rv is None and fail_on_unsat:
            print("Note: minimize failed with fail_on_unsat was true, resetting and retrying...")
                        
            glpk.glp_cpx_basis(self.lp) # resets the initial basis

            rv = self.minimize(direction_vec, columns, False)

            #LpInstance.print_verbose("Note: LP was infeasible, but then feasible after resetting statuses")

        if rv is None and fail_on_unsat:
            raise UnsatError("minimize returned UNSAT and fail_on_unsafe was True")

        return rv

    @staticmethod
    def get_simplex_error_string(simplex_res):
        '''get the error message when simplex() fails'''

        codes = [glpk.GLP_EBADB, glpk.GLP_ESING, glpk.GLP_ECOND, glpk.GLP_EBOUND, glpk.GLP_EFAIL, glpk.GLP_EOBJLL,
                 glpk.GLP_EOBJUL, glpk.GLP_EITLIM, glpk.GLP_ETMLIM, glpk.GLP_ENOPFS, glpk.GLP_ENODFS]

        msgs = [ \
            "Unable to start the search, because the initial basis specified " + \
            "in the problem object is invalid-the number of basic (auxiliary " + \
            "and structural) variables is not the same as the number of rows " + \
            "in the problem object.", 

            "Unable to start the search, because the basis matrix corresponding " + \
            "to the initial basis is singular within the working " + \
            "precision.",

            "Unable to start the search, because the basis matrix corresponding " + \
            "to the initial basis is ill-conditioned, i.e. its " + \
            "condition number is too large.",

            "Unable to start the search, because some double-bounded " + \
            "(auxiliary or structural) variables have incorrect bounds.",

            "The search was prematurely terminated due to the solver " + \
            "failure.",

            "The search was prematurely terminated, because the objective " + \
            "function being maximized has reached its lower " + \
            "limit and continues decreasing (the dual simplex only).",

            "The search was prematurely terminated, because the objective " + \
            "function being minimized has reached its upper " + \
            "limit and continues increasing (the dual simplex only).",

            "The search was prematurely terminated, because the simplex " + \
            "iteration limit has been exceeded.",

            "The search was prematurely terminated, because the time " + \
            "limit has been exceeded.",

            "The LP problem instance has no primal feasible solution " + \
            "(only if the LP presolver is used).",

            "The LP problem instance has no dual feasible solution " + \
            "(only if the LP presolver is used).",
            ]

        rv = "Unknown Error"

        for code, message in zip(codes, msgs):
            if simplex_res == code:
                rv = message
                break

        return rv

    def _process_simplex_result(self, simplex_res, columns):
        '''process the result of a glp_simplex call

        returns None on UNSAT, otherwise the optimization result with the requested columns
        if columns is None, will return full result
        '''

        rv = None

        if simplex_res == glpk.GLP_ENOPFS:  # no primal feasible w/ presolver
            rv = None
        elif simplex_res != 0: # simplex failed, report the error
            raise RuntimeError("glp_simplex returned nonzero status ({}): {}".format(
                simplex_res, LpInstance.get_simplex_error_string(simplex_res)))
        else:
            status = glpk.glp_get_status(self.lp)

            if status == glpk.GLP_NOFEAS: # infeasible
                rv = None
            elif status == glpk.GLP_OPT: # optimal
                lp_cols = self.get_num_cols()
                
                if columns is None:
                    rv = np.zeros(lp_cols)
                else:
                    rv = np.zeros(len(columns))

                # copy the output vars
                rv_len = len(rv)
                
                for i in range(rv_len):
                    col = i if columns is None else columns[i]

                    assert 0 <= col < lp_cols, "out of bounds column requested in LP solution: {}".format(col)

                    rv[i] = glpk.glp_get_col_prim(self.lp, int(col + 1))

            else: # neither infeasible nor optimal (for example, unbounded)
                codes = [glpk.GLP_OPT, glpk.GLP_FEAS, glpk.GLP_INFEAS, glpk.GLP_NOFEAS, glpk.GLP_UNBND, glpk.GLP_UNDEF]
                msgs = ["solution is optimal",
                        "solution is feasible",
                        "solution is infeasible",
                        "problem has no feasible solution",
                        "problem has unbounded solution",
                        "solution is undefined"]

                if status == glpk.GLP_UNBND:
                    ray = glpk.glp_get_unbnd_ray(self.lp)

                    raise RuntimeError(f"LP had unbounded solution in minimize(). Unbounded ray was variable #{ray}")

                for code, message in zip(codes, msgs):
                    if status == code:
                        raise RuntimeError("LP status after solving in minimize() was '{}': {}".format(message, code))

                raise RuntimeError("LP status after solving in minimize() was <Unknown>: {}".format(status))

        return rv

class UnsatError(RuntimeError):
    'raised if an LP is infeasible'

class SwigArray():
    '''Tracker for how much memoey swig arrays allocate (And leak, since there is a memory leak for these:
    see: https://github.com/biosustain/swiglpk/issues/31 )
    '''

    WARN_MEMORY_SWIGLPK_LEAK_GB = 4.0
    ERROR_MEMORY_SWIGLPK_LEAK_GB = 8.0
    bytes_allocated = 0

    @classmethod
    def as_double_array(cls, list_data):
        'wrapper for swig as_doubleArray'

        cls._allocated(8 * len(list_data))

        return glpk.as_doubleArray(list_data)

    @classmethod
    def as_int_array(cls, list_data):
        'wrapper for swig as_intArray'

        cls._allocated(8 * len(list_data))

        return glpk.as_intArray(list_data)

    @classmethod
    def _allocated(cls, num_bytes):
        'track how many bytes were allocated and print warning if threshold is exceeded'

        cls.bytes_allocated += num_bytes

        gb_warn = SwigArray.WARN_MEMORY_SWIGLPK_LEAK_GB
        warn_threshold = 1024**3 * gb_warn

        gb_error = SwigArray.ERROR_MEMORY_SWIGLPK_LEAK_GB
        error_threshold = 1024**3 * gb_error

        #print("Allocated: {} / {} ({:.2f}%)".format(
        #    cls.bytes_allocated, threshold, 100 * cls.bytes_allocated / threshold))

        if cls.bytes_allocated > warn_threshold:
            print(f"Swig array allocation leaked more than {gb_warn} GB memory. Warning limit can be raised by " + \
                "increasing lpinstance.SwigArray.WARN_MEMORY_SWIGLPK_LEAK_GB. For info on the leak, see: " + \
                  "https://github.com/biosustain/swiglpk/issues/31")

        if cls.bytes_allocated > error_threshold:
            raise MemoryError(
                f"Swig array allocation leaked more than {gb_error} GB memory. Error limit can be raised by " + \
                "increasing lpinstance.SwigArray.ERROR_MEMORY_SWIGLPK_LEAK_GB. For info on the leak, see: " + \
                  "https://github.com/biosustain/swiglpk/issues/31")
