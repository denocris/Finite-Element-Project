from numpy import *
from numpy.polynomial.legendre import leggauss
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import lagfunc as lf

import scipy.sparse.linalg


def FiniteElem3D(degree, dim, my_f):
    cheb = lf.chebyshev_nodes(degree+1) #Lista nodi chebichev

    n = degree + 1 # Dim poly space

    q,w = leggauss(n) # Gauss between -1 and 1
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

    # -------------------------------------------------

    latticeq_points = array([[[[qx,qy,qz] for qz in q] for qy in q ] for qx in q])
    latticeq_points = latticeq_points.reshape(len(q)*len(q)*len(q),dim)

    lpoint_x = array([latticeq_points[i,0] for i in range(len(latticeq_points))])
    lpoint_y = array([latticeq_points[i,1] for i in range(len(latticeq_points))])
    lpoint_z = array([latticeq_points[i,2] for i in range(len(latticeq_points))])

    # -------------------------------------------------
    print "Starting einsum A and M..."
    Atmp = einsum('jq, iq, q -> ji', Vpq, Vpq, w, optimize=True)
    Mtmp = einsum('jq, iq, q -> ji', Vq, Vq, w, optimize=True)

    A = einsum('il, jm, kn -> ijklmn', Atmp, Mtmp, Mtmp, optimize=True)
    A += einsum('il, jm, kn -> ijklmn', Mtmp, Atmp, Mtmp, optimize=True)
    A += einsum('il, jm, kn -> ijklmn', Mtmp, Mtmp, Atmp, optimize=True)

    M = einsum('il, jm, kn -> ijklmn', Mtmp, Mtmp, Mtmp, optimize=True)

    # -------------------------------------------------
    print "Starting einsum rhs..."
    my_f = array(my_f(lpoint_x,lpoint_y,lpoint_z)).reshape(n,n,n)

    rhs = einsum('ijk, li, i -> jkl', my_f, Vq, w, optimize=True)
    rhs = einsum('ijk, li, i -> jkl', rhs, Vq, w, optimize=True)
    rhs = einsum('ijk, li, i -> jkl', rhs, Vq, w, optimize=True)

    # -------------------------------------------------

    A = A.reshape(n**3,n**3)
    M = M.reshape(n**3,n**3)
    rhs = rhs.reshape(n**3)

    # ---------- Direct Method -------------------------
    print "Direct Method ..."

    u_fe_dir = linalg.solve( A + M, rhs)

    # ---------- Iterative Conjugate Gradient ----------
    print "Iteractive Method ..."

    P = diag( diag(A + M) ) # Preconditioner

    invP = linalg.inv(P)

    u_fe_iter = scipy.sparse.linalg.cg( A + M, rhs, M = invP)
    u_fe_iter = array(u_fe_iter[0])

    # -------------------------------------------------

    Vcheb = zeros((n, len(cheb)))

    for j in range(degree + 1):
        Vcheb[j] = lag_bas[j](cheb)

    C = einsum('is, jk, nm -> skmijn', Vcheb, Vcheb, Vcheb, optimize=True)

    sol_dir = einsum('skmijn, ijn', C, u_fe_dir.reshape((n, n, n)), optimize=True)
    sol_iter = einsum('skmijn, ijn', C, u_fe_iter.reshape((n, n, n)), optimize=True)

    return sol_dir.reshape(n,n,n), sol_iter.reshape(n,n,n)

if __name__ == "__main__":

    # Let us pick up this test function to test our solver

    u_exact = lambda x,y,z: cos(pi*x)*cos(pi*y)*cos(pi*z)
    my_f = lambda x,y,z: (3*(pi**2) + 1)*cos(pi*y)*cos(pi*x)*cos(pi*z)

    #--------- Plotting a section of our FE solution  --------

    dim = 3 # space dim of the problem
    degree = 8 # degree of polynomial bases

    u_fem_dir = FiniteElem3D(degree, dim, my_f)[0]

    cheby = lf.chebyshev_nodes(degree+1)
    X, Y = meshgrid(cheby,cheby)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X, Y, u_fem_dir[:,:,0], cmap = cm.jet)
    #ax.plot_surface(X, Y, u_fem_iter[:,:,0], cmap = cm.jet)
    #plt.show()

    # --------------- Error Computation ---------------------

    L2_err_dir = []
    L2_err_iter = []
    L2_err_match = []

    deg_start = 2
    deg_end = 18
    deg_step = 1

    for deg in range(deg_start, deg_end, deg_step):
        print "---------------------------------", deg
        u_ext_chebp = []

        cheb = lf.chebyshev_nodes(deg+1)

        u_fem_dir = FiniteElem3D(deg, dim, my_f)[0]
        u_fem_dir = u_fem_dir.reshape(len(cheb)**3,)

        u_fem_iter = FiniteElem3D(deg, dim, my_f)[1]
        u_fem_iter = u_fem_iter.reshape(len(cheb)**3,)

        for x in cheb:
            for y in cheb:
                for z in cheb:
                    u_ext_chebp.append(u_exact(x,y,z))

        u_ext_chebp = array(u_ext_chebp)

        L2_err_dir.append(linalg.norm(u_ext_chebp - u_fem_dir, ord=2))
        L2_err_iter.append(linalg.norm(u_ext_chebp - u_fem_iter, ord=2))
        L2_err_match.append(linalg.norm(u_fem_dir - u_fem_iter, ord=2))

    #print L2_err

    plt.figure()
    plt.title('FE L2 error plot')
    plt.semilogy(range(deg_start, deg_end, deg_step), L2_err_dir, 'b')
    plt.semilogy(range(deg_start, deg_end, deg_step), L2_err_iter, 'g')
    plt.xlabel('degree')
    plt.ylabel('L2 error')
    plt.show()
