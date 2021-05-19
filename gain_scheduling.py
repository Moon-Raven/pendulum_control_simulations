import numpy as np
from numpy import sin, cos, tan, arctan
import matplotlib.pyplot as plt
import scipy.integrate
import figure_creator as fc
from pathlib import Path

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

def simulation_func_modified(t, z, A, B, C, K1, K2, H, alpha, r):
    alpha = alpha(t)
    r = r(t)

    A, B, C = A(alpha), B(alpha), C(alpha)
    K1, K2, H = K1(alpha), K2(alpha), H(alpha)

    x1, x2 = z[0], z[1]
    eta = z[2]
    xhat1, xhat2 = z[3], z[4]

    x, xhat = np.array([x1, x2]), np.array([xhat1, xhat2])

    # Convert from np arrays to np matrices
    x, xhat = x[np.newaxis].T, xhat[np.newaxis].T
    B, C = B[np.newaxis].T, C[np.newaxis]
    K1, H = K1[np.newaxis], H[np.newaxis].T

    # Helper variables
    u = eta
    y = x2
    e = y - r

    x1_dot = tan(x1) + x2
    x2_dot = x1 + u
    y_dot = x2_dot
    eta_dot = -K1@xhat - K2*e
    xhat_dot = (A-B@K1-H@C)@xhat - B*K2*e + H*y_dot

    # Convert arrays to scalars 
    eta_dot = eta_dot[0,0]
    xhat1_dot = xhat_dot[0,0]
    xhat2_dot = xhat_dot[1,0]
    x2_dot = x2_dot

    z_dot = np.array([x1_dot, x2_dot, eta_dot, xhat1_dot, xhat2_dot])

    return z_dot

def simulate_modified(r, alpha, x0, t_start, t_end):
    """ Simulate example 12.6 using modified gain scheduling controller """
    A = lambda alpha : np.array([[1+alpha**2, 1], [1, 0]])
    B = lambda alpha : np.array([0, 1])
    C = lambda alpha : np.array([0, 1])
    H = lambda alpha : np.array([10+(1+alpha**2)*(4+alpha**2), 4+alpha**2])
    K11 = lambda x : (1+x**2)*(3+x**2) + 3 + 1/(1+x**2)
    K12 = lambda x : (3+x**2)
    K1 = lambda alpha : np.array([K11(alpha), K12(alpha)])
    K2 = lambda alpha : -1/(1+alpha**2)

    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    f = lambda t, x : simulation_func_modified(t, x, A, B, C, K1, K2, H, alpha, r)
    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    x1 = sol.y[0]
    x2 = sol.y[1]
    eta = sol.y[2]
    xhat1 = sol.y[3]
    xhat2 = sol.y[4]
    r = r(t)
    y = x2

    return t, x1, x2, eta, xhat1, xhat2, r, y

def simulate(r, alpha, x0, t_start, t_end):
    """ Simulate example 12.6 using unmodified gain scheduling controller """
    A = lambda alpha : np.array([[1+alpha**2, 1], [1, 0]])
    B = lambda alpha : np.array([0, 1])
    C = lambda alpha : np.array([0, 1])
    H = lambda alpha : np.array([10+(1+alpha**2)*(4+alpha**2), 4+alpha**2])
    K11 = lambda x : (1+x**2)*(3+x**2) + 3 + 1/(1+x**2)
    K12 = lambda x : (3+x**2)
    K1 = lambda alpha : np.array([K11(alpha), K12(alpha)])
    K2 = lambda alpha : -1/(1+alpha**2)

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
    y = x2

    return t, x1, x2, sigma, xhat1, xhat2, r, y

def main01_fixed_alpha():
    """ Simulate with control parameters fixed for alpha = 0 """
    x10, x20 = 0, 0
    sigma0 = 0
    xhat10, xhat20 = 0, 0
    x0 = np.array([x10, x20, sigma0, xhat10, xhat20])
    r = lambda t : 0.2 + (t>30)*0.2 + (t>60)*0.2
    alpha = lambda t : 0
    t_start, t_end = 0, 100

    t, x1, x2, sigma, xhat1, xhat2, r, y = simulate(r, alpha, x0, t_start, t_end)

    fig, ax = fc.new_figure()
    ax.plot(t, r, 'r--', label = '$r$')
    ax.plot(t, y, label = '$y$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/gain_scheduling_fixed_alpha2.pdf', pad_inches=0.0)

def main02_gain_scheduling_unmodified():
    """ Simulate with unmodified gain scheduling controller """
    x10, x20 = 0, 0
    sigma0 = 0
    xhat10, xhat20 = 0, 0
    x0 = np.array([x10, x20, sigma0, xhat10, xhat20])
    r = lambda t : 0.2 + (t>30)*0.2 + (t>60)*0.2
    alpha = lambda t : r(t)
    t_start, t_end = 0, 140

    t, x1, x2, sigma, xhat1, xhat2, r, y = simulate(r, alpha, x0, t_start, t_end)

    fig, ax = fc.new_figure()
    ax.plot(t, r, 'r--', label = '$r$')
    ax.plot(t, y, label = '$y$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/gain_scheduling_unmodified.pdf', pad_inches=0.0)

def main03_gain_scheduling_modified():
    """ Simulate with modified gain scheduling controller """
    x10, x20 = 0, 0
    eta0 = 0
    xhat10, xhat20 = 0, 0
    x0 = np.array([x10, x20, eta0, xhat10, xhat20])
    r = lambda t : 0.2 + (t>30)*0.2 + (t>60)*0.2 + (t>90)*0.2
    alpha = lambda t : r(t)
    t_start, t_end = 0, 140

    t,x1,x2,eta,xhat1,xhat2,r,y = simulate_modified(r, alpha, x0, t_start, t_end)

    fig, ax = fc.new_figure()
    ax.plot(t, r, 'r--', label = '$r$')
    ax.plot(t, y, label = '$y$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/gain_scheduling_modified.pdf', pad_inches=0.0)

def main04_gain_scheduling_modified_ramp():
    """ Simulate with modified gain scheduling controller """
    x10, x20 = 0, 0
    eta0 = 0
    xhat10, xhat20 = 0, 0
    x0 = np.array([x10, x20, eta0, xhat10, xhat20])
    r = lambda t : (t<100)*1/100*t + (t>=100)*1
    alpha = lambda t : r(t)
    t_start, t_end = 0, 120

    t,x1,x2,eta,xhat1,xhat2,r,y = simulate_modified(r, alpha, x0, t_start, t_end)

    fig, ax = fc.new_figure()
    ax.plot(t, r, 'r--', label = '$r$')
    ax.plot(t, y, label = '$y$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/gain_scheduling_modified_ramp.pdf', pad_inches=0.0)

def main05_gain_scheduling_modified_steep_ramp():
    """ Simulate with modified gain scheduling controller """
    x10, x20 = 0, 0
    eta0 = 0
    xhat10, xhat20 = 0, 0
    x0 = np.array([x10, x20, eta0, xhat10, xhat20])
    r = lambda t : (t<10)*1/10*t + (t>=10)*1
    alpha = lambda t : r(t)
    t_start, t_end = 0, 120

    t,x1,x2,eta,xhat1,xhat2,r,y = simulate_modified(r, alpha, x0, t_start, t_end)

    plt.plot(t, r, 'r--', label = '$r$')
    plt.plot(t, y, label = '$y$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend()
    plt.show()

    # fig, ax = fc.new_figure()
    # ax.plot(t, r, 'r--', label = '$r$')
    # ax.plot(t, y, label = '$y$')
    # ax.set_xlabel('$t$')
    # ax.grid()
    # ax.legend()

    # Path('./figures').mkdir(parents=True, exist_ok=True)
    # plt.savefig('figures/gain_scheduling_modified_steep_ramp.pdf', pad_inches=0.0)