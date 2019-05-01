# lpset
Lpset is a library for representing sets of states and geometric operations, using an linear programming (LP) formulation representation. The n-dimensional set of states is the orthogonal projection of all the LP variables (usually > n) onto n of the dimensions. Lpset uses a numerical linear programming solver (GLPK) and incremental construction (warm-start) of the LP constraints in order to support efficient set operations.

The code was initially extracted as the representation used by the Hylaa tool (https://github.com/stanleybak/hylaa).

# Installation
Lpset is written in python3. A .dockerfile is included with dependency installation instructions. You will also need to update PYTHONPATH to point to the parent of the lpset folder so you can use the library in your code.

# Set Operations
* Linear transformation, f(A, S): {Ax | x \in S}
* Minkowski sum, f(S1, S2): {x = a + b | a \in S1 and b \in S2}
* Minkowski difference, f(S1, S2): {x = a + b | a \in S1 and b \in S2}
* Convex hull, f(S1, S2): 

# Query Operations
* feasible query: does the set contain any points
* optimization query (support function): find the point in the set that maximizes the dot product with the given vector
