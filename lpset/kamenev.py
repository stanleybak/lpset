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
    _, _, vh = np.linalg.svd(a_mat)

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

    print(". checking hyperplane() result (todo: remove)")
    for pt in pts:
        assert np.allclose([np.dot(pt, normal_vec)], [rhs])

    return normal_vec, rhs
        
def verts(dims, supp_point_func, epsilon=1e-7):
    'get all the vertices of the set, in the given number of dimensions, defined through supp_point_func'

    tol = 1e-7
    rv = []

    init_simplex_verts = []
    interior_pt = np.zeros((dims,), dtype=float)

    for vec in regular_simplex_vecs(dims):
        pt = supp_point_func(vec)

        rv.append(pt)
        init_simplex_verts.append(pt)
        
        for i in range(dims):
            interior_pt[i] += pt[i]

    # interior_pt will be the centroid of the initial simplex vertices
    for i in range(dims):
        interior_pt[i] /= (dims + 1)

    print(f"init_simplex_verts: {init_simplex_verts}")
    print(f"interior_pt: {interior_pt}")

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
