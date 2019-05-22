'''
Functions related to Kamenev's method for polytope approximation (the method of refined bounds)

Stanley Bak
May 16, 2019
'''

import math
from heapq import heappush, heappop

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import lpplot3d
from util import Freezable
from timerutil import Timers

def regular_simplex_vecs(dims):
    '''get vertex points associated with a regular simplex in n-dimensions

    returns a list of dims+1 points, centered at the origin and equidistant from each other (not normalized)
    '''

    # this is based on the SIMPLEX_COORDINATES2 method (without the last normalization step):
    # https://people.sc.fsu.edu/~jburkardt/py_src/simplex_coordinates/simplex_coordinates.html   

    a = (1-math.sqrt(dims+1)) / dims

    # compute centroid
    centroid_sum = [a] * dims

    for identity_row in range(dims):
        centroid_sum[identity_row] += 1.0

    centroid = [x / (dims + 1.0) for x in centroid_sum]

    rv = []

    for row in range(dims):
        # construct each point from identity matrix
        pt = [1 if i == row else 0 for i in range(dims)]
        
        for col in range(dims):
            pt[col] -= centroid[col]

        rv.append(pt)

    # add last point from [a, a, a, ...]
    pt = [a - centroid[i] for i in range(dims)]
    rv.append(pt)

    return rv

def hyperplane(pts, interior_pt):
    '''construct the outward normal and rhs of a hyperplane passing through pts

    returns outward normal and rhs value; in 3-d we have ax + by + cz <= rhs, where <a, b, c> is the normal direction
    '''

    dims = len(pts[0])

    assert len(pts) == dims, "need n points each of which are n-dimensional to define a unique hyperplane"
    assert len(interior_pt) == dims, "point dimension mismatch"

    # potential problem: the points might not span the space... in this case any hyperplane is okay I guess?

    # we find the normal vector by constructing all vectors relative to the first point, doing a full SVD on them,
    # and then getting the last right singular vector (has a singular value of zero; so orthogonal to all vectors)

    a_mat = []
    pt0 = np.array(pts[0], dtype=float)

    for pt in pts[1:]:
        a_mat.append(np.array(pt, dtype=float) - pt0)

    # a_mat is now an (n-1) by n matrix
    a_mat = np.array(a_mat, dtype=float)
    a_mat.shape = (dims-1, dims)

    _, s, vh = np.linalg.svd(a_mat)

    tol = 1e-7
    assert len(s) == dims - 1 and (dims == 1 or s[-1] > tol), f"hyperplane was not unique , singular values: {s}"

    # the last singular vector should be orthogonal to all the vectors in a_mat
    normal_vec = vh[-1]

    rhs = np.dot(normal_vec, pts[0])

    # normalize
    norm = np.linalg.norm(normal_vec)
    assert np.allclose([norm], [1]), "norm should be 1"

    # use interior_pt to make sure we have the outward normal
    dot = np.dot(normal_vec, interior_pt)

    assert not np.allclose([dot], [rhs]), f"interior_point is on hyperplane ({normal_vec} = {rhs}); " + \
      "cannot deduce outward direction"
    
    if dot > rhs:
        normal_vec = -normal_vec
        rhs *= -1

    return normal_vec, rhs

def get_orthonormal_rank(vecs, tol=1e-7):
    '''
    given a list of vecs, return a new vector orthonormal to them and the rank of the matrix
    '''

    _, s, v = np.linalg.svd(vecs)

    index = 0

    while index < len(s) and s[index] > tol:
        index += 1

    if index == len(v):
        rv_vec = None # the vectors span the space
    else:
        rv_vec = v[index]

    return rv_vec, index

def get_rank(vecs, tol=1e-7):
    '''get the rank of the passed in matrix'''

    return get_orthonormal_rank(vecs, tol=tol)[1]

def find_two_points(dims, supp_point_func):
    '''find two points in the the convex set defined through supp_point_func (which may be degenerate)

    if len(pts) == 1, the convex set is a degenerate set consisting of a single pt
    '''

    pts = []

    for d in range(dims):
        vec = np.array([-1 if i == d else 0 for i in range(dims)], dtype=float)

        # try min
        p = supp_point_func(vec)

        if not pts:
            pts.append(p)
        elif not np.allclose(p, pts[0]):
            pts.append(p)
            break
        
        # try max
        vec = np.array([1 if i == d else 0 for i in range(dims)], dtype=float)
        p = supp_point_func(vec)

        if not np.allclose(p, pts[0]):
            pts.append(p)
            break

    return pts

def verts(dims, supp_point_func, epsilon=1e-7):
    '''
    get the n-dimensional vertices of the convex set defined through supp_point_func (which may be degenerate)
    '''

    Timers.tic('init_simplex')
    # first, construct the initial simplex and determine a basis for the convex set (it may be degenerate)
    pts = find_two_points(dims, supp_point_func)

    if len(pts) == 1: # S is a degenerate shape consisting of a single point
        print(f"find_two_points returned single point: {pts}")
        return pts
    
    init_simplex = [pts[0], pts[1]]

    init_vec = pts[1] - pts[0]
    print(f"two initial points: {pts}, init_vec: {init_vec}")
     
    spanning_dirs = [init_vec]
    degenerate_dirs = []
    vecs = [init_vec]

    for it in range(dims - 1):
        new_dir, rank = get_orthonormal_rank(vecs)
        print(f"\nIteration {it}: new orthonormal directon: {new_dir}")
        print(f"vecs was:\n{vecs}")

        # min/max in direction v, checking if it increases the rank of vecs
        pt = supp_point_func(new_dir)
        vecs.append(pt - init_simplex[0])

        if get_rank(vecs) > rank:
            print(f"rank increased after maximize, adding point {pt}")
            init_simplex.append(pt)
            spanning_dirs.append(vecs[-1])
            print(f"added new spanning direction: {vecs[-1]}")
            continue

        # rank did not increase with maximize, try minimize
        print(f"rank did not increase after maximize... trying minimize")
        vecs = vecs[0:-1] # pop vec

        pt = supp_point_func(-1 * new_dir)
        vecs.append(pt - init_simplex[0])

        if get_rank(vecs) > rank:
            print(f"rank increased after minimize, adding point {pt}")
            init_simplex.append(pt)
            spanning_dirs.append(vecs[-1])
            print(f"added new spanning direction: {vecs[-1]}")
            continue

        # rank still didn't increase, new_dir is orthogonal to shape S
        vecs = vecs[0:-1] # pop vec
        print(f"rank also didn't increase after minimize, polytope is degenerate in direction {new_dir}")

        vecs.append(new_dir) # forces a new orthonormal direction during the next iteration
        degenerate_dirs.append(new_dir)

    # spanning_dirs and degenerate_dirs form an orthonormal basis for R^n
    assert len(spanning_dirs) + len(degenerate_dirs) == dims
    assert len(init_simplex) == 1 + len(spanning_dirs)

    print(f"spanning_dirs:\n{spanning_dirs}")
    print(f"degenerate_dirs:\n{degenerate_dirs}")

    if len(degenerate_dirs) == 0:
        print(f"using identity as spanning dirs")
        spanning_dirs = np.identity(len(spanning_dirs), dtype=float)
    else:
        # convert spanning_dirs to an orthonnormal basis
        new_spanning_dirs = sp.linalg.orth(np.array(spanning_dirs, dtype=float).transpose())

        print(f"ortho spanning_dirs:\n{new_spanning_dirs}")

        assert len(spanning_dirs) == new_spanning_dirs.shape[1], "num cols changed during sp.linalg.orth()"
        spanning_dirs = new_spanning_dirs

    # len(spanning_dirs) is the dimensionality of the shape
    # degenerate_dirs can be used to compute an offset to be added to all final vertices
    #span_dims = len(spanning_dirs)
    
    #basis_mat = np.array(spanning_dirs + degenerate_dirs, dtype=float).transpose()
    #inv_basis_mat = np.linalg.inv(basis_mat)
    basis_mat = spanning_dirs
    inv_basis_mat = spanning_dirs.transpose().copy()

    # converting from projected space -> original space will use basis_mat
    # converting from original space -> projected space will use inv_basis_mat

    print(f"basis_mat:\n{basis_mat}")
    print(f"inv_basis_mat:\n{inv_basis_mat}")

    def modified_supp_pt_func(proj_vec):
        'supp_point_func defined in projected space'

        Timers.tic('supp_pt_func')

        #aug_proj_vec = [proj_vec[i] if i < len(proj_vec) else 0 for i in range(dims)]
        #orig_vec = np.dot(basis_mat, aug_proj_vec)
        orig_vec = np.dot(basis_mat, proj_vec)

        orig_pt = supp_point_func(orig_vec)

        # project proj_pt to spanning space
        proj_pt = np.dot(inv_basis_mat, orig_pt)

        Timers.toc('supp_pt_func')

        return proj_pt

    print(f"init_simplex_verts (original space): {init_simplex}")

    # project and truncate every point from init_simplex into the projected space
    proj_init_simplex = [np.dot(inv_basis_mat, vert) for vert in init_simplex]

    print(f"init_simplex_verts (projected space): {proj_init_simplex}")

    proj_dims = basis_mat.shape[1]

    Timers.toc('init_simplex')
    
    proj_verts = verts_given_init_simplex(proj_init_simplex, proj_dims, modified_supp_pt_func, epsilon=epsilon)

    # convert projected verts back to the original space
    rv = []

    null_space_offset = [np.dot(d, init_simplex[0]) * d for d in degenerate_dirs]

    for v in proj_verts:

        orig_pt = np.dot(basis_mat, v)

        # add null space offset
        for offset in null_space_offset:
            orig_pt += offset

        rv.append(orig_pt)

        print(f"proj_vert({v}) -> {orig_pt}")

    return rv

def centroid(pts):
    'get the centroid (average) of the given points'

    rv = np.zeros(pts[0].shape, dtype=float)

    for pt in pts:
        rv += pt

    rv /= len(pts)

    return rv

def plot_triangle_verts(ax, pts, color, lw=1):
    'plot triangle verties in 3d'

    assert len(pts[0]) == 3, f"verts dimension: {len(pts[0])}"
    assert len(pts) == 3, f"expected 3 verts per facet, got {len(pts)}"

    xs = [pts[-1][0]]
    ys = [pts[-1][1]]
    zs = [pts[-1][2]]

    for pt in pts:
        xs.append(pt[0])
        ys.append(pt[1])
        zs.append(pt[2])

    ax.plot(xs, ys, zs, color, lw=lw)

class Facet(Freezable):
    'facet data structure used when constructing verts'

    count = 0

    def __init__(self, verts, normal_vec, normal_rhs, epsilon, supporting_pt):
        self.epsilon = epsilon # error tolerance during construction

        # parallel lists: each neighbor facet shares a ridge with this facet if the corresponding vertex is excluded
        self.verts = verts
        self.neighbors = [None] * len(verts) # list of facets (parallel list to verts): should be populated later 

        assert np.allclose([np.linalg.norm(normal_vec)], [1.0])

        # these define the hyperplane
        self.normal_vec = normal_vec
        self.normal_rhs = normal_rhs

        # the supporting point in the set in the outward direction of this face.
        # If None, then face is extreme (within epsilon)
        # use set_support() to set
        self.supporting_pt = None
        self.supporting_val = None
        self.supporting_error = None

        self.facet_id = Facet.count
        Facet.count += 1

        # a flag used for deleting facets off the error-sorted heap, without actually touching the heap
        self.was_deleted = False

        self.set_support(supporting_pt)

        self.freeze_attrs()

    def __str__(self):
        return f"[Facet {self.facet_id} w/ error {self.supporting_error}, verts: {self.verts}]"

    def plot3d(self, ax, color='k:', lw=1):
        'plot this 3-d facet (2d triangle) on the given axis object'

        plot_triangle_verts(ax, self.verts, color, lw)

    def is_visible(self, pt):
        'is the passed in point visible from this facet?'

        rv = False
        
        val = np.dot(pt, self.normal_vec)
        error = val - self.normal_rhs

        if error > self.epsilon:
            rv = True

        return rv

    def set_support(self, supporting_pt):
        '''set the supporting point and value (point in the set S outside of this facet)

        if supporting_pt is within the hyperplane defined by this facet, this sets supporting_pt/val/error to None
        '''

        supporting_val = np.dot(supporting_pt, self.normal_vec)
        error = supporting_val - self.normal_rhs

        if error > self.epsilon:
            self.supporting_pt = supporting_pt
            self.supporting_val = supporting_val
            self.supporting_error = error
        else:
            self.supporting_pt = None
            self.supporting_val = None
            self.supporting_error = None

    def get_ridge(self, ridge_verts):
        '''is the passed-in ridge part of this facet?
        If so, return the exclude_vert_index that defines the ridge, else None
        '''

        rv_init = '<all were close?>' 
        rv = rv_init

        for index, vert in enumerate(self.verts):
            found_pt = False

            for pt in ridge_verts:
                if np.allclose(pt, vert):
                    found_pt = True
                    break

            if not found_pt:
                if rv == rv_init:
                    rv = index
                else:
                    rv = None # multiple indices not found... not a ridge
                    break

        assert rv != rv_init, "all verts were close to ridge verts; numerical precision issues probably"

        return rv

    
def max_error_facet(facets):
    'return the facet with the maximum error'

    # efficiency could be improved... this simply iterates over all facets

    rv = None

    for f in facets:
        if f.supporting_error is not None:
            if rv is None or rv.supporting_error < f.supporting_error:
                rv = f

    return rv

def get_visible_and_horizon(new_pt, start_facet):
    'get the visible facets and horizon ridges'

    visible_facets = [start_facet]
    horizon_ridges = [] # list of 2-tuples (facet, exclude_vert_index)

    unprocessed = [start_facet]

    while unprocessed:
        cur_facet = unprocessed.pop()

        for neb_facet in cur_facet.neighbors:
            assert neb_facet is not None

            if neb_facet in visible_facets: # already processed
                continue

            if neb_facet.is_visible(new_pt):
                visible_facets.append(neb_facet)
                unprocessed.append(neb_facet)
            else:
                # neb_facet is a horizon facet, find the ridge shared between neb_facet and cur_facet
                ridge_index = neb_facet.neighbors.index(cur_facet)

                horizon_ridges.append((neb_facet, ridge_index))

    assert horizon_ridges, "no horizon ridges?"

    return visible_facets, horizon_ridges

def make_new_facets(horizon_ridges, new_point, interior_pt, epsilon, supp_point_func):
    '''make and return new facets from a list of horizontal ridges and the new point
    '''

    rv = []

    for horizon_ridge in horizon_ridges:
        horizon_facet, exclude_vert_index = horizon_ridge

        #print(f"creating new facet from horizon ridge tuple: {horizon_ridge[0]}, {horizon_ridge[1]}")

        new_facet_verts = horizon_facet.verts.copy()
        del new_facet_verts[exclude_vert_index]
        new_facet_verts.append(new_point)

        #print(f"new facet verts: {new_facet_verts}")

        # construct hyperplane through verts
        new_normal, new_rhs = hyperplane(new_facet_verts, interior_pt)

        # evaluate the error on the given face
        new_supporting_pt = supp_point_func(new_normal)

        new_facet = Facet(new_facet_verts, new_normal, new_rhs, epsilon, new_supporting_pt)

        #print(f"adding new facet {new_facet}")
        rv.append(new_facet)

        # update neighbors with horizon_facet
        horizon_facet.neighbors[exclude_vert_index] = new_facet
        new_facet.neighbors[-1] = horizon_facet

    return rv

def assign_new_facet_neighbors(new_facets):
    'assign neighbors to newly-created facets'

    for new_facet in new_facets:
        #print(f"\nassigning neighbors to new facet: {new_facet}")

        for i, neighbor in enumerate(new_facet.neighbors):
            #print(f"processing new_facet.neighbor[{i}]: {neighbor}")

            if neighbor is not None:
                continue

            # construct the ridge
            Timers.tic('construct ridge')
            ridge = new_facet.verts.copy()
            del ridge[i]
            Timers.toc('construct ridge')

            #print(f"looking for neighbor with ridge: {ridge}")

            # look for the neighbor within new_facets
            for other_new_facet in new_facets:
                if other_new_facet is new_facet:
                    continue

                #print(f"checking neighbor {other_new_facet}")

                Timers.tic('get_ridge')
                exclude_index = other_new_facet.get_ridge(ridge)
                Timers.toc('get_ridge')


                if exclude_index is not None:
                    #print(f"found the ridge when excluding index {exclude_index}")

                    Timers.tic('assign_neb')
                    new_facet.neighbors[i] = other_new_facet
                    assert other_new_facet.neighbors[exclude_index] is None
                    other_new_facet.neighbors[exclude_index] = new_facet

                    Timers.toc('assign_neb')
                    break

            assert new_facet.neighbors[i] is not None, "neighbor of constructed new_facet not found"

def verts_given_init_simplex(init_simplex_verts, dims, supp_point_func, epsilon=1e-7):
    '''get all the vertices of the set, in the given number of dimensions, defined through supp_point_func

    This function is provided with an initial simplex which spans the space.
    '''

    assert len(init_simplex_verts) == dims + 1
    assert len(init_simplex_verts[0]) == dims

    print(f"init_simplex_verts: {init_simplex_verts}")

    rv = [] + init_simplex_verts

    Timers.tic('init_faces construction')
    init_facets = [] # initial facets, ordered by which vertex was excluded to construct them

    # construct facets from init_simplex_verts by excluding one of the vertices
    for exclude_index in range(dims+1):
        interior_pt = centroid(init_simplex_verts)
        pts = init_simplex_verts[:exclude_index] + init_simplex_verts[exclude_index + 1:]

        # construct hyperplane through pts
        f_normal, f_rhs = hyperplane(pts, interior_pt)

        # evaluate the error on the given face
        f_supporting_pt = supp_point_func(f_normal)

        f = Facet(pts, f_normal, f_rhs, epsilon, f_supporting_pt)

        init_facets.append(f)

    # facets which are on the convex hull
    extreme_facets = [] 

    # facets which are not part of the convex hull
    # heap of tuples: (-error, facet_id, facet)   [-error is used so maximum error is popped first]
    non_extreme_facet_heap = [] 

    # assign facet neighbors of init simplex
    for init_facet in init_facets:
        
        for exclude_vert_index in range(len(init_facet.verts)):
            # find the facet in init_facets which contains the ridge when exclude_vert_index is excluded from the facet

            if init_facet.neighbors[exclude_vert_index] is not None:
                continue # neighbor for this facet was already assigned

            # construct the ridge we're interested in:
            ridge = init_facet.verts.copy()
            del ridge[exclude_vert_index]

            # search the OTHER facets for this ridge
            for other_init_facet in init_facets:
                if other_init_facet is init_facet:
                    continue

                other_exclude_index = other_init_facet.get_ridge(ridge)

                if other_exclude_index is not None:
                    init_facet.neighbors[exclude_vert_index] = other_init_facet
                    assert other_init_facet.neighbors[other_exclude_index] is None
                    other_init_facet.neighbors[other_exclude_index] = init_facet
                    break

            assert init_facet.neighbors[exclude_vert_index] is not None, "neighbor of initial facet not found"

        # add the facet to either extreme_facets or non_extreme_facets
        if init_facet.supporting_error is None:
            extreme_facets.append(init_facet)
            print(f"init facet was extreme (on chull)")
        else:
            tup = (-init_facet.supporting_error, init_facet.facet_id, init_facet)
            heappush(non_extreme_facet_heap, tup)
            print(f"added non-extreme init facet with supporting error: {-tup[0]}")

    Timers.toc('init_faces construction')
    iteration = 0

    # process remaining facets one-by-one
    while non_extreme_facet_heap:
        _, _, max_f = heappop(non_extreme_facet_heap)

        iteration += 1
        print(f"\n-------")
        print(f"Iteration {iteration}; non-extreme-facets remaining: {1 + len(non_extreme_facet_heap)}")

        if max_f.was_deleted:
            print(f"max_facet was deleted, continuing")
            continue

        Timers.tic('loop')
        
        print(f"new_pt: {max_f.supporting_pt} from facet {max_f.facet_id} with error {max_f.supporting_error}")

        # extend the triangulation by removing newly-redundant facets after normal_supporting_pt was added
        # I believe this is from the "Beyond-Beneath" convex hull algorithm

        new_point = max_f.supporting_pt

        # TODO: remove this check that new_point not in rv
        for pt in rv:
            assert not np.allclose(pt, new_point), f"point was extreme twice: {new_point}"
        
        rv.append(new_point)

        interior_pt = centroid(max_f.verts + [new_point])

        Timers.tic('get_visible_and_horizon')
        visible_facets, horizon_ridges = get_visible_and_horizon(new_point, max_f)
        Timers.toc('get_visible_and_horizon')

        print(f"num visible_facets: {len(visible_facets)}")
        print(f"num horizon_ridges: {len(horizon_ridges)}")

        # mark all visible facets as deleted
        for vis_facet in visible_facets:
            print(f"deleting visible facet {vis_facet.facet_id}")
            vis_facet.was_deleted = True

        # create new facets from each of the horizon ridges
        # a horizon ridge consists of a tuple: (facet, exclude_vert_index)

        Timers.tic('make_new_facets')
        new_facets = make_new_facets(horizon_ridges, new_point, interior_pt, epsilon, supp_point_func)
        Timers.toc('make_new_facets')

        Timers.tic('assign_new_facet_neighbors')
        assign_new_facet_neighbors(new_facets)
        Timers.toc('assign_new_facet_neighbors')

        # add the facet to either extreme_facets or non_extreme_facets
        for new_facet in new_facets:
            if new_facet.supporting_error is None:
                print(f"new facet {new_facet.facet_id} was extreme")
                extreme_facets.append(new_facet)
            else:
                print(f"new facet {new_facet.facet_id} was NOT extreme... adding to heap for further processing")
                tup = (-new_facet.supporting_error, new_facet.facet_id, new_facet)
                heappush(non_extreme_facet_heap, tup)

        Timers.toc('loop')
        
    return np.array(rv, dtype=float)
