import numpy as np
from numpy import pi, sin, cos
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate
import pendulum
import figure_creator as fc
from pathlib import Path

def main01_constant_input_simulation():
    """ Simple simulation of a pendulum system. Choose a constant
        control input Tss, and simulate system's output behavior. """
    m, g, l, k = 1, 10, 1, 2
    pend = pendulum.Pendulum(m, g, l, k)

    theta0, theta_dot0 = 1, 0
    x0 = np.array([theta0, theta_dot0])

    t_start, t_end = 0, 10
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    # Constant control input
    Tss = 0
    T = lambda t, x : Tss
    f = lambda t, x : pend.state_space_static_controller(t, x, T)

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    fig, ax = fc.new_figure()
    ax.plot(sol.t, sol.y[0], label = '$\\theta$')
    ax.plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/unforced_response.pdf', pad_inches=0.0)

def main02_constant_input_simulation_fixed_delta():
    """ Simple simulation of a pendulum system. Choose a constant
        desired output angle delta; a constant control input Tss
        will be calculated to obtain a stable equilibrium at the
        desired output angle delta. """
    m, g, l, k = 1, 10, 1, 2
    a, b, c = g/l, k/m, 1/m/l/l
    pend = pendulum.Pendulum(m, g, l, k)

    theta0, theta_dot0 = 0, 0
    x0 = np.array([theta0, theta_dot0])

    t_start, t_end = 0, 10
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    # Constant control input derived from desired constant angle
    delta = pi/4
    Tss = a/c * sin(delta)
    T = lambda t, x : Tss
    f = lambda t, x : pend.state_space_static_controller(t, x, T)

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    fig, ax = fc.new_figure()
    ax.plot([t_start, t_end], [delta, delta], 'r--', label='Reference $\\delta$')
    ax.plot(sol.t, sol.y[0], label = '$\\theta$')
    ax.plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/constant_control.pdf', pad_inches=0.0)

def main03_feedback_state_stabilization():
    """ Simulation of a pendulum system. The system is controlled by
        a stabilizing state-feedback controller around desired angle
        delta. """
    m, g, l, k = 1, 10, 1, 2
    a, b, c = g/l, k/m, 1/m/l/l
    delta = pi/4 # Desired value of angle theta at equilibrium point
    pend = pendulum.Pendulum(m, g, l, k)

    theta0 , theta_dot0 = 0, 0
    x0 = np.array([theta0, theta_dot0])

    # State feedback control
    k1, k2 = 1, 1
    T = lambda t, x: -k1*(x[0]-delta) - k2*x[1] + a*sin(delta)/c
    f = lambda t, x: pend.state_space_static_controller(t, x, T)

    t_start, t_end = 0, 10
    N = 1000
    t = np.linspace(t_start, t_end, N)
    tspan = [t_start, t_end]

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)
    control = T(sol.t, sol.y)

    fig, axs = fc.new_figure(subplot_count=2, height=2*fc.DEFAULT_HEIGHT)
    axs[0].plot([t_start, t_end], [pi/4, pi/4], 'r--', label='Reference $\\delta$')
    axs[0].plot(sol.t, sol.y[0], label = '$\\theta$')
    axs[0].plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    axs[1].plot(sol.t, control, 'g', label = '$T$')
    axs[0].set_xlabel('$t$')
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel('$t$')
    axs[1].grid()
    axs[1].legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/state_feedback.pdf', pad_inches=0.0)

def main04_output_feedback_stabilization():
    """ Simulation of a pendulum system. The system is controlled by
        a stabilizing output-feedback controller around desired angle
        delta. States are estimated by an observer. """
    m, g, l, k = 1, 10, 1, 2
    delta = pi/4
    pend = pendulum.Pendulum(m, g, l, k)

    x0 = np.array([0, 0, 0, 0])
    t_start, t_end = 0, 10
    N = 1000
    t = np.linspace(t_start, t_end, N)
    tspan = [t_start, t_end]

    k1, k2, h1, h2 = 1, 1, 1, 1
    K = np.array([k1, k2])
    H = np.array([h1, h2])

    f_original = pend.state_space_observer_controller
    f = lambda t, x : f_original(t, x, K, H, delta)

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    fig, ax = fc.new_figure()
    ax.plot([t_start, t_end], [delta, delta], 'r--', label='Reference $\\delta$')
    ax.plot(sol.t, sol.y[0], label = '$\\theta$')
    ax.plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    ax.plot(sol.t, sol.y[2], label = '$\\hat{\\theta}$')
    ax.plot(sol.t, sol.y[3], label = '$\\hat{\\dot{\\theta}}$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/output_feedback.pdf', pad_inches=0.0)

def test_integral_controller_constraints(K, m, g, l, k, delta):
    """ Test whether the stabilization constraints for the integral
        controller are satisfied. """
    a, b, c = g/l, k/m, 1/m/l/l
    k1, k2, k3 = K[0], K[1], K[2]

    constraint1 = b + k2*c
    constraint2 = k3 * c
    constraint3 = (b + k2*c) * (a*cos(delta) + k1*c) - k3*c

    print('Constraint values:')
    print(f'\tConstraint 1: {constraint1}')
    print(f'\tConstraint 2: {constraint2}')
    print(f'\tConstraint 3: {constraint3}')

    print('Final result: ', end='')
    if constraint1 < 0 or constraint2 < 0 or constraint3 < 0:
        print('Constraints violated!')
    else:
        print('Constraints fulfilled!')

def main05_integral_control():
    """ Simulation of a pendulum system. The system is controlled by
        an integral controller, which is stabilized by linearization
        around desired angle delta and state-feedback. """
    # Good values for m to test are: 0.1, 1, 2 (1 is nominal)
    m, g, l, k = 0.1, 10, 1, 2
    pend = pendulum.Pendulum(m, g, l, k)
    delta = pi/4
    k1, k2, k3 = 8, 2, 10
    K = np.array([k1, k2, k3])
    test_integral_controller_constraints(K, m, g, l, k, delta)

    theta0, theta_dot0, sigma0 = 0, 0, 0
    x0 = np.array([theta0, theta_dot0, sigma0])

    t_start, t_end = 0, 10
    N = 1000
    t = np.linspace(t_start, t_end, N)
    tspan = [t_start, t_end]

    f_original = pend.state_space_integral_controller
    f = lambda t, x : f_original(t, x, K, delta)

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    fig, ax = fc.new_figure()
    ax.plot([t_start, t_end], [delta, delta], 'r--', label='Reference $\\delta$')
    ax.plot(sol.t, sol.y[0], label = '$\\theta$')
    ax.plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    ax.plot(sol.t, sol.y[2], label = '$\\sigma$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/integral_control_small_mass.pdf', pad_inches=0.0)