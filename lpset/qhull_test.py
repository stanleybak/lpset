'''
qhull scalability test
'''

import random
import time

import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt

def main():
    'main entry point'

    dims = 4
    num_verts = 2**dims

    verts = rand(dims, num_verts)

    #new_pt = [1] * dims # corner
    new_pt = [1] + [0] * (dims - 1) # edge

    verts.append(new_pt)
    
    #verts = cube(dims)

    # QG<index> - exclude index in points list, save visible facets to hull.good
    # Qv<index> Qg Q0 - only construct facets to include hull.good
    
    print(f"dims: {dims}, len(verts): {len(verts)}")
    start = time.time()
    hull = ConvexHull(verts, qhull_options=f"QG{len(verts) - 1} Qg")
    print(f"Runtime: {time.time() - start}")
    
    print(f"hull.points: {len(hull.points)}; hull.simplices: {len(hull.simplices)}")
    print(f"hull.good?: {'<None>' if hull.good is None else sum(hull.good)}")

    print(hull.simplices)
    

    plot = False
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        points = np.array(verts, dtype=float)
        ax.plot(points[:, 0], points[:, 1], 'o')

        if hull.good is not None:
            for visible_facet in hull.simplices[hull.good]:
                ax.plot(hull.points[visible_facet, 0],
                        hull.points[visible_facet, 1],
                        color='violet', lw=5)


        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'k-')

        plt.show()

    #new_pt = [1] * dims # corner
    #new_pt = [1] + [0] * (dims - 1) # edge

    # equations for hyperplane:
    # (hull.equations[i,:-1] * coord).sum() + hull.equations[i,-1] == 0

    # rand summary
    # 2 dims: 4 simplices
    # 3 dims: 12 simplices 
    # 4 dims: 58 simplices
    # ...
    # 6 dims: 0.014 secs, 3443 simplices
    # 7 dims: 0.29 secs, 37564 simplices
    # 8 dims: 6 secs, 460281 simplices
    # ----------------
    # more verts
    # 6 dims (1280 verts): 1.3 secs, 168722 simplics 

    # cube summary
    # 2 dims: 4 simplices
    # 3 dims: 12 simplices
    # 4 dims: 58 simplices 
    # ...
    # 7 dims: 0.08 secs, 12532 simplices 
    # 8 dims: 1.5 secs, 82924 simplices 
    # 9 dims: 70 secs, 870244 simplices

def rand(dims, pts):
    'return verts of random points on unit ball'

    random.seed(0)

    verts = []

    for _ in range(pts):
        vec = np.array([random.random() - 0.5 for _ in range(dims)], dtype=float)
        vec = vec / np.linalg.norm(vec)

        verts.append(vec)

    return verts

def cube(dims):
    'return verts of unit cube'

    verts = []

    for it in range(2**dims):
        cur = it

        vert = []

        for _ in range(dims):
            if cur % 2 == 0:
                vert.append(1)
            else:
                vert.append(0)

            cur //= 2

        verts.append(vert)

    return verts

if __name__ == '__main__':
    main()
