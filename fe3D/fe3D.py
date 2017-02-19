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

    VVV    = einsum('ij,kl,nm -> inkljm', Vq, Vq, Vq)
    VVVp   = einsum('ij,kl,nm -> inkljm', Vq, Vq, Vpq)
    VVpV   = einsum('ij,kl,nm -> inkljm', Vq, Vpq, Vq)
    VpVV   = einsum('ij,kl,nm -> inkljm', Vpq, Vq, Vq)
    VpVpVp = einsum('ij,kl,nm -> inkljm', Vpq, Vpq, Vpq)

    VVV  = reshape(VVV,  (prod(VVV.shape[:dim]),  prod(VVV.shape[dim:])))
    VVVp = reshape(VVVp, (prod(VVVp.shape[:dim]), prod(VVVp.shape[dim:])))
    VVpV = reshape(VVpV, (prod(VVpV.shape[:dim]), prod(VVpV.shape[dim:])))
    VpVV = reshape(VpVV, (prod(VpVV.shape[:dim]), prod(VpVV.shape[dim:])))

    W = einsum('i,j,k -> ijk', w, w, w)
    W = reshape(W, (prod(W.shape[:dim])))

    latticeq_points = np.array([[[[qx,qy,qz] for qz in q] for qy in q ] for qx in q])
    latticeq_points = latticeq_points.reshape(len(q)*len(q)*len(q),dim)

    lpoint_x = np.array([latticeq_points[i,0] for i in range(len(latticeq_points))])
    lpoint_y = np.array([latticeq_points[i,1] for i in range(len(latticeq_points))])
    lpoint_z = np.array([latticeq_points[i,2] for i in range(len(latticeq_points))])

    # -------------------------------------------------

    A = einsum('jq, iq, q -> ij', VVVp, VVVp, W)
    A += einsum('jq, iq, q -> ij', VVpV, VVpV, W)
    A += einsum('jq, iq, q -> ij', VpVV, VpVV, W)

    M = einsum('jq, iq, q -> ij', VVV, VVV, W)

    # -------------------------------------------------

    rhs = einsum('iq, q, q -> i', VVV, W, my_f(lpoint_x,lpoint_y,lpoint_z))

    # -------------------------------------------------

    u_fe = linalg.solve( A + M, rhs)

    Vcheb = zeros((n, len(cheb)))

    for j in range(degree + 1):
        Vcheb[j] = V[j](cheb)

    C = einsum('is, jk, nm -> skmijn', Vcheb, Vcheb, Vcheb)

    sol = einsum('skmijn, ijn', C, u_fe.reshape((n, n, n)))

    return sol.reshape(n,n,n)

if __name__ == "__main__":

    dim = 3 # space dim of the problem
    degree = 5 # degree of polynomial bases

    # Let us pick up this function to test our solver

    u_exact = lambda x,y,z: cos(pi*x)*cos(pi*y)*cos(pi*z)
    my_f = lambda x,y,z: (3*(pi**2) + 1)*cos(pi*y)*cos(pi*x)*cos(pi*z)

    u_fem = solver2D(degree, dim, my_f)

    #print u_fem


    #--------- Plotting Finite Element Solution --------

    cheb = lf.chebyshev_nodes(degree+1)

    X, Y = meshgrid(cheb,cheb)

    my_col = cm.jet(Z/np.amax(Z))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,u_fem[:,:,1], cmap=cm.jet)
    plt.show()
