from numpy import *
from numpy.polynomial.legendre import leggauss
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import lagfunc as lf

import scipy.sparse.linalg


def solver_cg_preconditioner_3D(degree, dim, my_f):
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


    Vq = zeros((n, len(q)))
    Vpq = zeros((n, len(q)))

    #Le righe di Vq sono le funzioni di base calcolate sui punti di quadratura
    for i in range(n):
        Vq[i] = lag_bas[i](q)
        Vpq[i] = lag_bas_deriv[i](q)

    VVV    = einsum('ij,kl,nm -> inkljm', Vq, Vq, Vq, optimize=True)
    VVVp   = einsum('ij,kl,nm -> inkljm', Vq, Vq, Vpq, optimize=True)
    VVpV   = einsum('ij,kl,nm -> inkljm', Vq, Vpq, Vq, optimize=True)
    VpVV   = einsum('ij,kl,nm -> inkljm', Vpq, Vq, Vq, optimize=True)

    VVV  = reshape(VVV,  (prod(VVV.shape[:dim]),  prod(VVV.shape[dim:])))
    VVVp = reshape(VVVp, (prod(VVVp.shape[:dim]), prod(VVVp.shape[dim:])))
    VVpV = reshape(VVpV, (prod(VVpV.shape[:dim]), prod(VVpV.shape[dim:])))
    VpVV = reshape(VpVV, (prod(VpVV.shape[:dim]), prod(VpVV.shape[dim:])))

    W = einsum('i,j,k -> ijk', w, w, w, optimize=True)
    W = reshape(W, (prod(W.shape[:dim])))

    latticeq_points = array([[[[qx,qy,qz] for qz in q] for qy in q ] for qx in q])
    latticeq_points = latticeq_points.reshape(len(q)*len(q)*len(q),dim)

    lpoint_x = array([latticeq_points[i,0] for i in range(len(latticeq_points))])
    lpoint_y = array([latticeq_points[i,1] for i in range(len(latticeq_points))])
    lpoint_z = array([latticeq_points[i,2] for i in range(len(latticeq_points))])

    # -------------------------------------------------

    A = einsum('jq, iq, q -> ij', VVVp, VVVp, W, optimize=True)
    A += einsum('jq, iq, q -> ij', VVpV, VVpV, W, optimize=True)
    A += einsum('jq, iq, q -> ij', VpVV, VpVV, W, optimize=True)

    M = einsum('jq, iq, q -> ij', VVV, VVV, W, optimize=True)


    # -------------------------------------------------

    rhs = einsum('iq, q, q -> i', VVV, W, my_f(lpoint_x,lpoint_y,lpoint_z), optimize=True)

    # ---------- Iterative Conjugate Gradient ---------------

    P = diag( diag(A + M) ) # Preconditioner

    invP = linalg.inv(P)

    u_fe = scipy.sparse.linalg.cg( A + M, rhs, M = invP)

    Vcheb = zeros((n, len(cheb)))

    for j in range(degree + 1):
        Vcheb[j] = lag_bas[j](cheb)

    u_fe = array(u_fe[0])

    C = einsum('is, jk, nm -> skmijn', Vcheb, Vcheb, Vcheb, optimize=True)

    sol = einsum('skmijn, ijn', C, u_fe.reshape((n, n, n)), optimize=True)

    return sol.reshape(n,n,n)

if __name__ == "__main__":

    dim = 3 # space dim of the problem
    degree = 5 # degree of polynomial bases

    # Let us pick up this function to test our solver

    u_exact = lambda x,y,z: cos(pi*x)*cos(pi*y)*cos(pi*z)
    my_f = lambda x,y,z: (3*(pi**2) + 1)*cos(pi*y)*cos(pi*x)*cos(pi*z)

    u_fem = solver_cg_preconditioner_3D(degree, dim, my_f)

    #print u_fem


    #--------- Plotting Finite Element Solution --------

    cheby = lf.chebyshev_nodes(degree+1)

    X, Y = meshgrid(cheby,cheby)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X, Y, u_fem[:,:,0], cmap = cm.jet)
    #plt.show()

    # --------------- Error Computation ---------------------

    L2_err = []

    deg_start = 2
    deg_end = 6
    deg_step = 1

    for deg in range(deg_start, deg_end, deg_step):
        u_ext_chebp = []

        cheb = lf.chebyshev_nodes(deg+1)

        u_fem = solver_cg_preconditioner_3D(deg, dim, my_f)
        u_fem = u_fem.reshape(len(cheb)**3,)

        for x in cheb:
            for y in cheb:
                for z in cheb:
                    u_ext_chebp.append(u_exact(x,y,z))

        u_ext_chebp = array(u_ext_chebp)

        L2_err.append(linalg.norm(u_ext_chebp - u_fem, ord=2))
        print "---------------------------------", deg

    print L2_err

    plt.figure()
    plt.title('FE Iteractive Method - L2 error plot')
    #plt.semilogy(range(2,6), L2_err)
    plt.loglog(range(deg_start, deg_end, deg_step), L2_err)
    plt.xlabel('degree')
    plt.ylabel('L2 error')
    plt.show()
