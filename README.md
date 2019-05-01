# Lpset
Lpset is a python3 library for representing sets of states and geometric operations, using an linear programming (LP) formulation representation. The n-dimensional set of states is the orthogonal projection of all the LP variables (usually > n) onto n of the dimensions. Lpset uses a numerical linear programming solver (GLPK) and incremental construction (warm-start) of the LP constraints in order to efficiently implement set operations. In theory we could use an exact LP solver to get exact results, but that is not done here (maybe in the future if this is useful). Note that incremental construction means that most of the operations modify the set in place, and this is usually more efficient than making copies of objects.

The code was initially extracted from the set representation used by the Hylaa tool (https://github.com/stanleybak/hylaa). The development was done by Stanley Bak (http://stanleybak.com).

# Installation
Lpset is written in python3. A `.dockerfile` is included with dependency installation instructions. Remember to update `PYTHONPATH` to point to the parent of the `lpset` folder in order to use the library in your code.

# Set Operations
* **Linear Transformation**, f(A, S): {Ax | x \in S}
* **Minkowski Sum**, f(S1, S2): {a + b | a \in S1, b \in S2}
* **Minkowski (Pontryagin/Geometric) Difference**, f(S1, S2): {c | c + B \subseteq A}
* **Convex Hull**, f(S1, S2): {(x)a + (1-x)b | a \in S1, b \in S2, 0 <= x <= 1}
* **Intersection**, f(S1, S2): {x | x \in S1 and x \in S2}

# Query Operations
* **Feasible**: does the set contain any points?
* **Optimization** (support function): find the point in the set that maximizes the dot product with the given vector?
* **Point Containtment**: is a given point in the set?
* **2D Approx Vertices** (plotting): gets a list of vertices of the (approximate) set projected onto two dimensions

# Lpset Construction Options
* **n-d Box**, construct an lpset given interval bounds for each dimension
* **Zonotope**, construct an lpset from a center and list of generators
* **Linear Constraints**, construct an lpset from an arbitrary list of linear constriants
* **Lpset**, clone or deep copy an existing lpset
