
from numpy import *
from numpy.polynomial.polynomial import *



def chebyshev_nodes(x):
    cheb = [0.5*cos((2*(i+1)-1)*pi/(2*x)) + 0.5 for i in range(x)]
    cheb = array(cheb)
    return sort(cheb)

def lagrange_basis(xi, i):
    def func(x):
        assert i<len(xi) and i>=0, 'Out of range: 0 < i < len(xi)'
        ret = 1;
        for xj in xi[range(i)+range(i+1,len(xi))]:
            p = (x-xj)/(xi[i]-xj)
            ret *= p
        return ret
    return func

def lagrange_basis_derivatives(xi, i):
    def func(x):
        deriv_sum=0
        assert i<len(xi) and i>=0, 'Out of range: 0 < i < len(xi)'
        ret = 1;
        for xj in xi[range(i)+range(i+1,len(xi))]:
            p = ((x-xj)/(xi[i]-xj))
            ret *= p
        for xj in xi[range(i)+range(i+1,len(xi))]:
            deriv_sum += 1./(x-xj)
        ret *= deriv_sum
        return ret
    return func

def lagrange_function(v,lag_base):
    def func(x):
        res=0
        for i in range(len(v)):
            res += v[i]*lag_base[i](x)
        return res
    return func
