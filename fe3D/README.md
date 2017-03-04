##### Finite Element in 3D

The main files are:

- fe3D.py (normal implementation)
- fe3D_matfree.py (matrix free implementation)
- lagfunc.py (module containing lagrangian bases)

They are structured in order to run on Ulysses, where the updated version of numpy and scipy was installed.

Part of the early work was done on jupiter notebook ('fed3D.ipynb').

All the relevant results and plots are shown in the file:

- plotting.ipynb

By default the right end side of the problem is:
$$
f(x, y, z) = cos(\pi x) cos(\pi y) cos(\pi z)
$$
