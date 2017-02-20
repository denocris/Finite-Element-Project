from numpy import *
from numpy.polynomial.legendre import leggauss
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import lagfunc as lf


def solver2D(degree, dim, my_f):
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
    # dual_basis_points = linspace(0,1,degree+1)
    #
    # for node in dual_basis_points:
    #     N.append(lambda f, node=node : f(node))

    #--------------------------------------------------------------------

    # Matrix for the change of variables
    # C = zeros((n,n))
    # for i in range(n):
    #     for j in range(n):
    #         C[i,j] = N[i](lag_bas[j])
    #
    #
    # for k in range(n):
    #     ei = zeros((n,))
    #     ei[k] = 1. # delta_ik
    #     vk = linalg.solve(C, ei)
    #     V.append(lf.lagrange_function(vk,lag_bas))
    #     V_prime.append(lf.lagrange_function(vk,lag_bas_deriv))

    # Now we evaluate all local basis functions and all derivatives of the basis functions at the quadrature points.

    Vq = zeros((n, len(q)))
    Vpq = zeros((n, len(q)))

    #Le righe di Vq sono le funzioni di base calcolate sui punti di quadratura
    for i in range(n):
        Vq[i] = lag_bas[i](q)
        Vpq[i] = lag_bas_deriv[i](q)

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

    rhs = einsum('iq, q, q -> i', VqVq, W, my_f(lpoint_x,lpoint_y)) #inserita

    # -------------------------------------------------

    u_fe = linalg.solve((A + M), rhs)

    Vcheb = zeros((n, len(cheb)))

    for j in range(degree + 1):
        Vcheb[j] = lag_bas[j](cheb)

    C = einsum('is, jk -> skij', Vcheb, Vcheb)

    sol = einsum('skij, ij', C, u_fe.reshape(n, n))

    return sol.reshape(n**2,)

if __name__ == "__main__":

    dim = 2 # space dim of the problem
    degree = 3 # degree of polynomial bases

    # Let us pick up this function to test our solver

    u_exact = lambda x,y: cos(pi*x)*cos(pi*y)
    my_f = lambda x,y: (2*(pi**2) + 1)*cos(pi*y)*cos(pi*x)

    u_fem = solver2D(degree, dim, my_f)


    #--------- Plotting Finite Element Solution --------

    cheb = lf.chebyshev_nodes(degree+1)

    X, Y = meshgrid(cheb,cheb)
    u_fem = u_fem.reshape((degree+1,degree+1))
    print u_fem

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,u_fem)
    #plt.show()

    # --------------- Error Computation ---------------------

    #max_err = []
    L2_err = []

    for deg in range(2,26):
        u_ext_chebp = []

        cheb = lf.chebyshev_nodes(deg+1)

        u_fem = solver2D(deg, dim, my_f)

        for i in cheb:
            for j in cheb:
                u_ext_chebp.append(u_exact(i,j))

        #max_err.append(linalg.norm(u_ext_chebp - u_fem, ord=inf))
        L2_err.append(linalg.norm(u_ext_chebp - u_fem, ord=2))
        print "---------------------------------", deg


    fig = plt.figure()
    plt.semilogy(range(2,26), L2_err)
    plt.show()
