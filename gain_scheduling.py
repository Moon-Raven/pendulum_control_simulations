import numpy as np
from numpy import sin, cos, tan, arctan
import matplotlib.pyplot as plt
import scipy.integrate

def simulation_func(t, z, A, B, C, K1, K2, H, alpha, r):
    alpha = alpha(t)
    r = r(t)

    A, B, C = A(alpha), B(alpha), C(alpha)
    K1, K2, H = K1(alpha), K2(alpha), H(alpha)

    x1, x2 = z[0], z[1]
    sigma = z[2]
    xhat1, xhat2 = z[3], z[4]

    x, xhat = np.array([x1, x2]), np.array([xhat1, xhat2])

    # Convert from np arrays to np matrices
    x, xhat = x[np.newaxis].T, xhat[np.newaxis].T
    B, C = B[np.newaxis].T, C[np.newaxis]
    K1, H = K1[np.newaxis], H[np.newaxis].T

    u = -K1 @ xhat - K2 * sigma

    x1_dot = tan(x1) + x2
    x2_dot = x1 + u
    sigma_dot = x2 - r
    xhat_dot = A@xhat + B @ (-K1@xhat - K2*sigma) + H @ (C@x - C@xhat)

    xhat1_dot, xhat2_dot = xhat_dot[0], xhat_dot[1]

    # Convert arrays to scalars 
    xhat1_dot = xhat1_dot[0]
    xhat2_dot = xhat2_dot[0]
    x2_dot = x2_dot[0,0]

    z_dot = np.array([x1_dot, x2_dot, sigma_dot, xhat1_dot, xhat2_dot])

    return z_dot

def simulate(r, alpha, x0):
    x10, x20 = 0, 0
    sigma0 = 0
    xhat10, xhat20 = 0, 0
    x0 = np.array([x10, x20, sigma0, xhat10, xhat20])

    A = lambda alpha : np.array([[1+alpha**2, 1], [1, 0]])
    B = lambda alpha : np.array([0, 1])
    C = lambda alpha : np.array([0, 1])
    H = lambda alpha : np.array([10+(1+alpha**2)*(4+alpha**2), 4+alpha**2])
    K11 = lambda x : (1+x**2)*(3+x**2) + 3 + 1/(1+x**2)
    K12 = lambda x : (3+x**2)
    K1 = lambda alpha : np.array([K11(alpha), K12(alpha)])
    K2 = lambda alpha : -1/(1+alpha**2)

    r = lambda t : 0.2 + 0*t
    alpha = lambda t : 0

    t_start, t_end = 0, 70
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    f = lambda t, x : simulation_func(t, x, A, B, C, K1, K2, H, alpha, r)
    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    x1 = sol.y[0]
    x2 = sol.y[1]
    sigma = sol.y[2]
    xhat1 = sol.y[3]
    xhat2 = sol.y[4]
    r = r(t)

def main01_fixed_alpha():
    """ Perform simulation with control parameters fixed for alpha = 0 """

    plt.plot(t, r, 'r--', label = '$r$')
    plt.plot(t, x1, label = '$x_1$')
    plt.plot(t, x2, label = '$x_2$')
    plt.plot(t, sigma, label = '$\sigma$')
    plt.plot(t, xhat1, label = '$\hat{x_1}$')
    plt.plot(t, xhat2, label = '$\hat{x_2}$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend()
    plt.show()
