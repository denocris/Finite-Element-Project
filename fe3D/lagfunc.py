
from numpy import *
from numpy.polynomial.polynomial import *
from numpy.polynomial.chebyshev import chebgauss


def chebyshev_nodes(num_nodes):
    #cheb = [0.5*cos((2*(i+1)-1)*pi/(2*x)) + 0.5 for i in range(num_nodes)]
    #cheb = array(cheb)
    cheb = 0.5 * chebgauss(num_nodes)[0] + 0.5
    return sort(cheb)

def lagrange_basis(nodes, i):
    nr = arange(len(nodes))
    d = product([ (nodes[i]-nodes[j]) for j in nr if j != i], axis=0)
    def func(x):
        L = product([ (x - nodes[j]) for j in nr if j != i], axis=0) / d
        return L
    return func

def lagrange_basis_derivatives(nodes, i):
    nr = arange(len(nodes))
    d = product([ (nodes[i]-nodes[j]) for j in nr if j != i], axis=0)
    def func(x):
        DL = sum( [ product( [ (x - nodes[j]) for j in nr if (j != i and j != k) ],axis=0) for k in nr if k != i ], axis=0 ) / d
        return DL
    return func


# def lagrange_basis(q,i):
#     n = len(q)
#     L = Polynomial.fromroots([xj for xj in q if xj != q[i]])
#     L = L * ( 1 / L(q[i]) )
#     return L
#
# def lagrange_basis_derivatives(q,i):
#     L_deriv = lagrange_basis(q,i).deriv()
#     return L_deriv
