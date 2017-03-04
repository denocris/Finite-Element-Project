##### Finite Element in 3D

The main files are:

- fe3D.py (normal implementation)
- fe3D_matfree.py (matrix free implementation)
- lagfunc.py (module containing lagrangian bases)

They are structured in order to run on Ulysses, where the updated version of numpy and scipy was installed.

By default the right end side of the problem is:
$$
f(x, y, z) = cos(\pi x) cos(\pi y) cos(\pi z)
$$

A simplified version of the code for the direct and iteractive method (without matrix free) can be seen in the jupiter notebook 'fed3D.ipynb'. In particular here is contained a surface plot of a section of the solution

All the relevant results and plots are shown in the file:

- plotting.ipynb
