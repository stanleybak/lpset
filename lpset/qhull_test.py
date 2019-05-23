'''
qhull scalability test
'''

import random
import time

import numpy as np
from scipy.spatial import ConvexHull

def main():
    'main entry point'

    dims = 6

    verts = rand(dims, 20 * 2**dims)
    #verts = cube(dims)
    
    print(f"dims: {dims}, len(verts): {len(verts)}")
    start = time.time()
    hull = ConvexHull(verts)
    print(f"simplices: {len(hull.simplices)}; runtime: {time.time() - start}")

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
