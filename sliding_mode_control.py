import numpy as np
import matplotlib.pyplot as plt
import timeseries_integrator
import scipy.integrate
from scipy.integrate import trapz
import figure_creator as fc
from pathlib import Path

def main01_check_matrices():
    A = np.array([[2.3, 0, 1],
                  [-4.9, 3, -3],
                  [-1, 0, 1]])
    B = np.array([[0.2, -1, 1],
                  [-0.9, -1, 0],
                  [-0.2, 0.1, 0]])
    D = np.array([[0], [0], [1]])
    K = np.array([[1.6, -1]])

    A11 = A[0:2,0:2]
    A12 = A[0:2,2:3]
    A21 = A[2:3,0:2]
    A22 = A[2:3,2:3]
    B11 = B[0:2,0:2]
    B12 = B[0:2,2:3]
    B21 = B[2:3,0:2]
    B22 = B[2:3,2:3]
    E11 = A11 + B11
    E12 = A12 + B12
    M = E11 - E12 @ K

    print(f'A11:')
    print(A11)
    print(f'A12:')
    print(A12)
    print(f'A21:')
    print(A21)
    print(f'A22:')
    print(A22)
    print(f'B11:')
    print(B11)
    print(f'B12:')
    print(B12)
    print(f'B21:')
    print(B21)
    print(f'B22:')
    print(B22)
    print('M:')
    print(M)
    w, v = np.linalg.eig(M)
    print('Eigenvalues of M:')
    print(w)

class Parameters:
    def __init__(self, A, B, D, tau, K):
        self.A = A
        self.B = B
        self.D = D
        self.tau = tau
        self.K = K

        self.A11 = A[0:2,0:2]
        self.A12 = A[0:2,2:3]
        self.A21 = A[2:3,0:2]
        self.A22 = A[2:3,2:3]
        self.B11 = B[0:2,0:2]
        self.B12 = B[0:2,2:3]
        self.B21 = B[2:3,0:2]
        self.B22 = B[2:3,2:3]
        self.E11 = self.A11 + self.B11
        self.E12 = self.A12 + self.B12
        self.M = self.E11 - self.E12 @ self.K

def simulation_func(t, z, v, tsi, par):
    # Update tsis
    tsi[0].add_sample(t, z[0])
    tsi[1].add_sample(t, z[1])
    tsi[2].add_sample(t, z[2])

    # Transform from np.array to np.matrix (column vector)
    z = z[np.newaxis].T

    # Fetch delayed states
    z_delayed1 = tsi[0].get_y(t-par.tau)
    z_delayed2 = tsi[1].get_y(t-par.tau)
    z_delayed3 = tsi[2].get_y(t-par.tau)
    z_delayed = np.array([[z_delayed1], [z_delayed2], [z_delayed3]])

    # Calculate derivatives of states at current time
    z_dot = par.A@z + par.B@z_delayed + par.D*v
    return z_dot[:,0]

def control_law(t, z, tsi, par, g):
    D_inv = 1

    # Transform from np.array to np.matrix (column vector)
    z1 = z[0:2][np.newaxis].T
    z2 = z[2:3][np.newaxis].T
    s = z2 + par.K@z1

    # Fetch delayed states
    z_delayed1 = tsi[0].get_y(t-par.tau)
    z_delayed2 = tsi[1].get_y(t-par.tau)
    z_delayed3 = tsi[2].get_y(t-par.tau)

    z1_delayed = np.array([z_delayed1, z_delayed2])[np.newaxis].T
    z2_delayed = z_delayed3

    # Calculate main terms of control law
    term1 = par.A21@z1 + par.B21@z1_delayed + par.A22*z2 + par.B22*z2_delayed
    term2 = par.K @ (par.A11@z1 + par.B11@z1_delayed + par.A12*z2 + par.B12*z2_delayed)
    term3 = g * np.sign(s)

    # Finalize control law
    v = D_inv * (-term1 -term2 -term3)
    v = v[0,0]
    return v

def dynamic_g(t, z, tsi, par):
    c = 0.4
    z1 = z[0:2]
    tmin = t - par.tau
    tmax = t
    integration_timespan = [tmin, tmax]
    t_arr, z11 = tsi[0].get_integration_arrays(integration_timespan)
    t_arr, z12 = tsi[1].get_integration_arrays(integration_timespan)
    z1_norm = np.abs(z11) + np.abs(z12)
    integral = trapz(z1_norm, t_arr)
    g = np.linalg.norm(z1, 1) + c * integral
    return g

def recreate_control(t_arr, z_arr, v):
    N = len(t_arr)
    v_arr = np.ones(N) * np.nan

    for i in range(N):
        t = t_arr[i]
        z = z_arr[:,i]
        v_arr[i] = v(t, z)

    return v_arr

def main02_simulate_fixed_g():
    A = np.array([[2.3, 0, 1],
                  [-4.9, 3, -3],
                  [-1, 0, 1]])
    B = np.array([[0.2, -1, 1],
                  [-0.9, -1, 0],
                  [-0.2, 0.1, 0]])
    D = np.array([[0], [0], [1]])
    tau = 0.1
    K = np.array([[1.6, -1]])
    g = 1
    par = Parameters(A, B, D, tau, K)
    z10, z20, z30 = 2, 2, 2
    z0 = np.array([z10, z20, z30])
    tsi1 = timeseries_integrator.TimeseriesIntegrator()
    tsi1.add_sample(0, z10)
    tsi2 = timeseries_integrator.TimeseriesIntegrator()
    tsi2.add_sample(0, z20)
    tsi3 = timeseries_integrator.TimeseriesIntegrator()
    tsi3.add_sample(0, z30)
    tsi = [tsi1, tsi2, tsi3]
    v = lambda t, z : control_law(t, z, tsi, par, g)
    f = lambda t, z : simulation_func(t, z, v(t,z), tsi, par)

    tmin, tmax = 0, 10
    tspan = [tmin, tmax]
    sample_count = 1000
    t = np.linspace(tmin, tmax, sample_count)

    sol = scipy.integrate.solve_ivp(f, tspan, z0, t_eval=t)

    z = sol.y
    z1 = z[0]
    z2 = z[1]
    z3 = z[2]
    v = recreate_control(t, z, v)

    fig, axs = fc.new_figure(subplot_count=2, height=2*fc.DEFAULT_HEIGHT)
    axs[0].plot(t, z1, label='$z_1$')
    axs[0].plot(t, z2, label='$z_2$')
    axs[0].plot(t, z3, label='$z_3$')
    axs[0].set_xlabel('$t$')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(t, v, label='$v$')
    axs[1].set_xlabel('$t$')
    axs[1].grid()
    axs[1].legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/smc_fixed_g.pdf', pad_inches=0.0)

def main03_simulate_dynamic_g_small_tau():
    A = np.array([[2.3, 0, 1],
                  [-4.9, 3, -3],
                  [-1, 0, 1]])
    B = np.array([[0.2, -1, 1],
                  [-0.9, -1, 0],
                  [-0.2, 0.1, 0]])
    D = np.array([[0], [0], [1]])
    tau = 0.1
    K = np.array([[1.6, -1]])
    par = Parameters(A, B, D, tau, K)
    z10, z20, z30 = 2, 2, 2
    z0 = np.array([z10, z20, z30])
    tsi1 = timeseries_integrator.TimeseriesIntegrator()
    tsi1.add_sample(0, z10)
    tsi2 = timeseries_integrator.TimeseriesIntegrator()
    tsi2.add_sample(0, z20)
    tsi3 = timeseries_integrator.TimeseriesIntegrator()
    tsi3.add_sample(0, z30)
    tsi = [tsi1, tsi2, tsi3]

    g = lambda t, z : dynamic_g(t, z, tsi, par)
    v = lambda t, z : control_law(t, z, tsi, par, g(t, z))
    f = lambda t, z : simulation_func(t, z, v(t,z), tsi, par)

    tmin, tmax = 0, 10
    tspan = [tmin, tmax]
    sample_count = 1000
    t = np.linspace(tmin, tmax, sample_count)

    sol = scipy.integrate.solve_ivp(f, tspan, z0, t_eval=t)

    z = sol.y
    z1 = z[0]
    z2 = z[1]
    z3 = z[2]
    v = recreate_control(t, z, v)

    fig, axs = fc.new_figure(subplot_count=2, height=2*fc.DEFAULT_HEIGHT)
    axs[0].plot(t, z1, label='$z_1$')
    axs[0].plot(t, z2, label='$z_2$')
    axs[0].plot(t, z3, label='$z_3$')
    axs[0].set_xlabel('$t$')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(t, v, label='$v$')
    axs[1].set_xlabel('$t$')
    axs[1].grid()
    axs[1].legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/smc_dynamic_g_small_tau.pdf', pad_inches=0.0)

def main04_simulate_dynamic_g_large_tau():
    A = np.array([[2.3, 0, 1],
                  [-4.9, 3, -3],
                  [-1, 0, 1]])
    B = np.array([[0.2, -1, 1],
                  [-0.9, -1, 0],
                  [-0.2, 0.1, 0]])
    D = np.array([[0], [0], [1]])
    tau = 0.35
    K = np.array([[1.6, -1]])
    par = Parameters(A, B, D, tau, K)
    z10, z20, z30 = 2, 2, 2
    z0 = np.array([z10, z20, z30])
    tsi1 = timeseries_integrator.TimeseriesIntegrator()
    tsi1.add_sample(0, z10)
    tsi2 = timeseries_integrator.TimeseriesIntegrator()
    tsi2.add_sample(0, z20)
    tsi3 = timeseries_integrator.TimeseriesIntegrator()
    tsi3.add_sample(0, z30)
    tsi = [tsi1, tsi2, tsi3]

    g = lambda t, z : dynamic_g(t, z, tsi, par)
    v = lambda t, z : control_law(t, z, tsi, par, g(t, z))
    f = lambda t, z : simulation_func(t, z, v(t,z), tsi, par)

    tmin, tmax = 0, 10
    tspan = [tmin, tmax]
    sample_count = 1000
    t = np.linspace(tmin, tmax, sample_count)

    sol = scipy.integrate.solve_ivp(f, tspan, z0, t_eval=t)

    z = sol.y
    z1 = z[0]
    z2 = z[1]
    z3 = z[2]
    v = recreate_control(t, z, v)

    fig, axs = fc.new_figure(subplot_count=2, height=2*fc.DEFAULT_HEIGHT)
    axs[0].plot(t, z1, label='$z_1$')
    axs[0].plot(t, z2, label='$z_2$')
    axs[0].plot(t, z3, label='$z_3$')
    axs[0].set_xlabel('$t$')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(t, v, label='$v$')
    axs[1].set_xlabel('$t$')
    axs[1].grid()
    axs[1].legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/smc_dynamic_g_large_tau.pdf', pad_inches=0.0)