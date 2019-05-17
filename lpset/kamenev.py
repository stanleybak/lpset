'''
Functions related to Kamenev's method for polytope approximation (the method of refined bounds)

Stanley Bak
May 16, 2019
'''

import math
from heapq import heappush, heappop

import numpy as np

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
    _, s, vh = np.linalg.svd(a_mat)

    print(f"singular values: {s}")

    tol = 1e-7
    assert len(s) == dims - 1 and s[-1] > tol, f"hyperplane was not uniquely defined, singular values: {s}"

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
        elif not np.allclose(p, p[0]):
            pts.append(p)
            break
        
        # try max
        vec = np.array([1 if i == d else 0 for i in range(dims)], dtype=float)
        p = supp_point_func(vec)

        if not np.allclose(p, p[0]):
            pts.append(p)
            break

    return pts

def verts(dims, supp_point_func, epsilon=1e-7):
    '''
    get the n-dimensional vertices of the convex set defined through supp_point_func (which may be degenerate)
    '''

    # first, construct the initial simplex and determine a basis for the convex set (it may be degenerate)
    pts = find_two_points(dims, supp_point_func)

    if len(pts) == 1: # S is a degenerate shape consisting of a single point
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

    # len(spanning_dirs) is the dimensionality of the shape
    # degenerate_dirs can be used to compute an offset to be added to all final vertices
    span_dims = len(spanning_dirs)
    
    basis_mat = np.array(spanning_dirs + degenerate_dirs, dtype=float).transpose()
    inv_basis_mat = np.linalg.inv(basis_mat)

    # converting from projected space -> original space will use basis_mat
    # converting from original space -> projected space will use inv_basis_mat

    print(f"basis_mat:\n{basis_mat}")
    print(f"inv_basis_mat:\n{inv_basis_mat}")

    def modified_supp_pt_func(proj_vec):
        'supp_point_func defined in projected space'

        aug_proj_vec = [proj_vec[i] if i < len(proj_vec) else 0 for i in range(dims)]
        orig_vec = np.dot(basis_mat, aug_proj_vec)

        orig_pt = supp_point_func(orig_vec)

        # project proj_pt to spanning space
        proj_pt = np.dot(inv_basis_mat, orig_pt)
        trun_proj_pt = proj_pt[0:span_dims]

        return trun_proj_pt

    print(f"init_simplex_verts (original space): {init_simplex}")

    # project and truncate every point from init_simplex into the projected space
    proj_init_simplex = [np.dot(inv_basis_mat, vert)[0:span_dims] for vert in init_simplex]

    print(f"init_simplex_verts (projected space): {proj_init_simplex}")

    proj_verts = verts_given_init_simplex(proj_init_simplex, len(spanning_dirs), modified_supp_pt_func, epsilon=epsilon)

    # convert projected verts back to the original space
    rv = []

    null_space_offset = [np.dot(degenerate_dir, init_simplex[0]) for degenerate_dir in degenerate_dirs]

    for v in proj_verts:
        full_d_pt = [v[i] if i < len(v) else null_space_offset[i - span_dims] for i in range(dims)]
                
        pt = np.dot(basis_mat, full_d_pt)

        rv.append(pt)

        print(f"proj_vert({v}) -> {pt}")

    return rv

def verts_given_init_simplex(init_simplex_verts, dims, supp_point_func, epsilon=1e-7):
    '''get all the vertices of the set, in the given number of dimensions, defined through supp_point_func

    This function is provided with an initial simplex which spans the space.
    '''

    assert len(init_simplex_verts) == dims + 1
    assert len(init_simplex_verts[0]) == dims

    tol = 1e-7
    interior_pt = np.zeros((dims,), dtype=float)

    for pt in init_simplex_verts:
        interior_pt += pt

    interior_pt /= (dims + 1)

    print(f"init_simplex_verts: {init_simplex_verts}")
    print(f"interior_pt: {interior_pt}")

    rv = [] + init_simplex_verts

    # priority queue key is a 2-tuple, (-1 * error, count) (-1 so maximum error gets popped first, count to break ties)
    # value is a 4-tuple corresponding to a single facet: (verts, hyperplane_normal, hyperplane_rhs, supporting_pt)
    remaining_facets_heap = []
    heap_count = 0

    # construct facets from init_simplex_verts by excluding one of the vertices
    for exclude_index in range(dims+1):
        pts = init_simplex_verts[:exclude_index] + init_simplex_verts[exclude_index + 1:]

        # construct hyperplane through pts
        normal, rhs = hyperplane(pts, interior_pt)

        # evaluate the error on the given face
        supporting_pt = supp_point_func(normal)
        support_val = np.dot(supporting_pt, normal)

        error = support_val - rhs

        print(f"pts for facet: {pts}")
        print(f"equation for hyperplane: ({normal})*x <= {rhs}")
        print(f"supporting_pt in normal direction: {supporting_pt}")
        print(f"supporting_val: {support_val}")

        assert error + tol >= 0, "support_val < hyperplane_rhs? shouldn't happen for convex sets"

        if error > epsilon:
            key = (-error, heap_count)
            heap_count += 1
            val = (pts, normal, rhs, supporting_pt)
            heappush(remaining_facets_heap, (key, val))
            print(f"added facet with error {error} to heap")

    iteration = 0
    
    # process remaining faces one-by-one
    while remaining_facets_heap:
        obj = heappop(remaining_facets_heap)
        neg_error = obj[0][0]

        iteration += 1
        print(f"\n-------")
        print(f"Iteration {iteration}: cur_error: {-neg_error}, cur_remaining_facets: {len(remaining_facets_heap)}")
        facet_verts, normal, rhs, normal_supporting_pt = obj[1]

        print(f"facet_verts: {facet_verts}")
        print(f"rv verts: {rv}")
        print(f"normal_supporting_pt: {normal_supporting_pt}")

        rv.append(normal_supporting_pt)

        #interior_pt = np.zeros((dims,), dtype=float)
        #
        #for pt in facet_verts:
        #    interior_pt += pt
        #
        #interior_pt /= len(facet_verts)

        # split this facet into n new facets, by excluding a single point from verts and including supp_pt
        assert len(facet_verts) == dims

        for exclude_index in range(dims):
            pts = facet_verts[:exclude_index] + facet_verts[exclude_index + 1:]
            pts.append(normal_supporting_pt)

            print(f"\nexcluding index {exclude_index}")
            print(f"pts for facet: {pts}")
            print(f"interior_pt: {interior_pt}")

            # construct hyperplane through pts
            normal, rhs = hyperplane(pts, interior_pt)

            # evaluate the error on the given face
            supporting_pt = supp_point_func(normal)
            support_val = np.dot(supporting_pt, normal)

            error = support_val - rhs

            print(f"equation for hyperplane: ({normal})*x <= {rhs}")
            print(f"supporting_pt in normal direction: {supporting_pt}")
            print(f"supporting_val: {support_val}")
            print(f"error: {error}")

            assert error + tol >= 0, "support_val < hyperplane_rhs? shouldn't happen for convex sets"

            if error > epsilon:
                key = (-error, heap_count)
                heap_count += 1
                val = (pts, normal, rhs, supporting_pt)
                heappush(remaining_facets_heap, (key, val))
               
                print(f"added facet with error {error} to heap. Heap size: {len(remaining_facets_heap)}")
            
    return np.array(rv, dtype=float)
