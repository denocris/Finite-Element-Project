from numpy import *
from numpy.polynomial.legendre import leggauss
import lagfunc as lf


def solver2D(degree, dim):
    cheb = lf.chebyshev_nodes(degree+1) #Lista nodi chebichev

    n = degree + 1 # Dim poly space

    num_q = 2 * degree + 1
    q,w = leggauss(num_q) # Gauss between -1 and 1
    q = (q+1)/2 # to go back to 0,1
    w = w/2

    lag_bas=[]
    lag_bas_deriv=[]
    N = []
    V = []
    V_prime=[]

    #lagrangian base per i punti chebichev
    for i in range(len(cheb)):
        lag_bas.append(lf.lagrange_basis(cheb,i))
        lag_bas_deriv.append(lf.lagrange_basis_derivatives(cheb,i))

    #--------------------------------------------------------------------
    #punti dello spazio duale dove vai a calcolare le funzioni di base# dim = degree +1
    dual_basis_points = linspace(0,1,degree+1)

    for node in dual_basis_points:
        N.append(lambda f, node=node : f(node))

    #--------------------------------------------------------------------

    # Matrix for the change of variables
    C = zeros((n,n))
    for i in range(n):
        for j in range(n):
            C[i,j] = N[i](lag_bas[j])


    for k in range(n):
        ei = zeros((n,))
        ei[k] = 1. # delta_ik
        vk = linalg.solve(C, ei)
        V.append(lf.lagrange_function(vk,lag_bas))
        V_prime.append(lf.lagrange_function(vk,lag_bas_deriv))

    # Now we evaluate all local basis functions and all derivatives of the basis functions at the quadrature points.

    Vq = zeros((n, len(q)))
    Vpq = zeros((n, len(q)))

    #Le righe di Vq sono le funzioni di base calcolate sui punti di quadratura
    for i in range(n):
        Vq[i] = V[i](q)
        Vpq[i] = V_prime[i](q)

    VqVq   = einsum('ij,kl -> ikjl', Vq,  Vq)
    VqVpq  = einsum('ij,kl -> ikjl', Vq,  Vpq)
    VpqVq  = einsum('ij,kl -> ikjl', Vpq, Vq)
    VpqVpq = einsum('ij,kl -> ikjl', Vpq, Vpq)


    VqVq   = reshape(VqVq,   (prod(VqVq.shape[:dim]),   prod(VqVq.shape[dim:])))
    VqVpq  = reshape(VqVpq,  (prod(VqVpq.shape[:dim]),  prod(VqVpq.shape[dim:])))
    VpqVq  = reshape(VpqVq,  (prod(VpqVq.shape[:dim]),  prod(VpqVq.shape[dim:])))
    VpqVpq = reshape(VpqVpq, (prod(VpqVpq.shape[:dim]), prod(VpqVpq.shape[dim:])))

    W = einsum('i,j->ij',w,w)
    W = reshape(W, (prod(W.shape[:dim])))


    latticeq_points = array([[[qx,qy] for qy in q] for qx in q ])
    latticeq_points = latticeq_points.reshape(len(q)*len(q),2)

    lpoint_x = array([latticeq_points[i,0] for i in range(len(latticeq_points))])
    lpoint_y = array([latticeq_points[i,1] for i in range(len(latticeq_points))])

    # -------------------------------------------------

    A = einsum('jq, iq, q -> ij', VqVpq, VqVpq, W)
    A += einsum('jq, iq, q -> ij', VpqVq, VpqVq, W)

    M = einsum('jq, iq, q -> ij', VqVq, VqVq, W)

    # -------------------------------------------------

    myf_test = lambda x,y: (2*(pi**2) + 1)*cos(pi*y)*cos(pi*x)

    rhs = einsum('iq, q, q -> i', VqVq, W, myf_test(lpoint_x,lpoint_y)) #inserita

    # -------------------------------------------------

    return linalg.solve((A + M), rhs)

if __name__ == "__main__":

    dim = 2 # space dim of the problem
    degree = 3 # degree of polynomial bases

    ufem = solver2D(degree, dim)

    print ufem
