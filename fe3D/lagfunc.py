
from numpy import *
from numpy.polynomial.polynomial import *
from numpy.polynomial.chebyshev import chebgauss


def chebyshev_nodes(num_nodes):
    #cheb = [0.5*cos((2*(i+1)-1)*pi/(2*x)) + 0.5 for i in range(num_nodes)]
    #cheb = array(cheb)
    cheb = 0.5 * chebgauss(num_nodes)[0] + 0.5
    return sort(cheb)


def lagrange_basis(q,i):
    n = len(q)
    L = Polynomial.fromroots([xj for xj in q if xj != q[i]])
    L = L * ( 1 / L(q[i]) )
    return L

def lagrange_basis_derivatives(q,i):
    L_deriv = lagrange_basis(q,i).deriv()
    return L_deriv

def lagrange_function(v,lag_base):
    def func(x):
        res=0
        for i in range(len(v)):
            res += v[i]*lag_base[i](x)
        return res
    return func
