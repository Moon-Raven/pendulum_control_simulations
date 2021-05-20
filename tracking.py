import numpy as np
from numpy import sin, cos, tan, arctan, pi
import matplotlib.pyplot as plt
import scipy.integrate
import figure_creator as fc
import pendulum as pend
from pathlib import Path

def reference_model(t, x, w, omega, zeta):
    """ Ordinary differential equations for reference modelling """
    y1 = x[0]
    y2 = x[1]
    w = w(t)

    y1_dot = y2
    y2_dot = -np.power(omega,2)*y1 - 2*zeta*omega*y2 + np.power(omega,2)*w

    y_dot = np.array([y1_dot, y2_dot])
    return y_dot

def main01_reference_model():
    """ Test reference modeling """
    x0 = np.array([0, 0])
    omega = 10
    zeta = 1

    w = lambda t : 1.5 + (t>2)*(-1.5)
    t_start, t_end = 0, 3
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    f = lambda t, x : reference_model(t, x, w, omega, zeta)
    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)
    w = w(t)
    r = sol.y[0]

    fig, ax = fc.new_figure()
    ax.plot(t, w, label = '$w$')
    ax.plot(t, r, label = '$r$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()

    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/tracking_reference_model.pdf', pad_inches=0.0)

def pendulum_simple_simfunc(t, alpha, rm_func, w, a, b, c, u):
    # Collect inputs
    x, y = alpha[0:2], alpha[2:4]
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]

    # Evaluate reference model
    y_dot = rm_func(t, y, w)

    # Evaluate helper signals
    r, r_dot = y1, y2
    r_dotdot = y_dot[1]
    e1, e2 = x1-r, x2-r_dot
    e = np.array([e1, e2])
    u = u(t, x, e, r_dotdot)

    # Evaluate pendulum
    x1_dot = x2
    x2_dot = -a*sin(x1) - b*x2 + c*u
    x_dot = np.array([x1_dot, x2_dot])

    # Collect results
    alpha_dot = np.concatenate((x_dot, y_dot))
    return alpha_dot

def main02_pendulum_simple():
    """ First tracking pendulum example """
    x10, x20, y10, y20 = 0, 0, 0, 0
    x0 = np.array([x10, x20, y10, y20])
    omega, zeta = 10, 1
    a, b, c = 10, 1, 10
    k1, k2 = 400, 20

    w = lambda t : 1.5 + (t>2)*(-1.5)
    t_start, t_end = 0, 3
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    u = lambda t, x, e, r_dd: 1/c * (a*sin(x[0]) + b*x[1] + r_dd - k1*e[0] - k2*e[1])
    rm_func = lambda t, x, w : reference_model(t, x, w, omega, zeta)
    f = lambda t, x : pendulum_simple_simfunc(t, x, rm_func, w, a, b, c, u)
    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    # Collect outputs
    w = w(t)
    r = sol.y[2]
    y = sol.y[0]

    fig, ax = fc.new_figure()
    ax.plot(t, r, label = '$r$')
    ax.plot(t, y, '--', label = '$y$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()
    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/tracking_pendulum1.pdf', pad_inches=0.0)

def main03_pendulum_simple_initial_cond():
    """ First tracking pendulum example, non-trivial initial conditions """
    x10, x20, y10, y20 = 0.8, 0, 0, 0
    x0 = np.array([x10, x20, y10, y20])
    omega, zeta = 10, 1
    a, b, c = 10, 1, 10
    k1, k2 = 400, 20

    w = lambda t : 1.5 + (t>2)*(-1.5)
    t_start, t_end = 0, 3
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    u = lambda t, x, e, r_dd: 1/c * (a*sin(x[0]) + b*x[1] + r_dd - k1*e[0] - k2*e[1])
    rm_func = lambda t, x, w : reference_model(t, x, w, omega, zeta)
    f = lambda t, x : pendulum_simple_simfunc(t, x, rm_func, w, a, b, c, u)
    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)
    # Collect outputs
    w = w(t)
    r = sol.y[2]
    y = sol.y[0]

    fig, ax = fc.new_figure()
    ax.plot(t, r, label = '$r$')
    ax.plot(t, y, '--', label = '$y$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()
    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/tracking_pendulum2.pdf', pad_inches=0.0)

def main04_pendulum_simple_perturbed():
    """ First tracking pendulum example, perturbed parameters """
    x10, x20, y10, y20 = 0, 0, 0, 0
    x0 = np.array([x10, x20, y10, y20])
    omega, zeta = 10, 1
    a, b, c = 10, 1, 10
    b2, c2 = 0.5, 5
    k1, k2 = 400, 20

    w = lambda t : 1.5 + (t>2)*(-1.5)
    t_start, t_end = 0, 3
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    u = lambda t, x, e, r_dd: 1/c * (a*sin(x[0]) + b*x[1] + r_dd - k1*e[0] - k2*e[1])
    rm_func = lambda t, x, w : reference_model(t, x, w, omega, zeta)
    f = lambda t, x : pendulum_simple_simfunc(t, x, rm_func, w, a, b2, c2, u)
    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    # Collect outputs
    w = w(t)
    r = sol.y[2]
    y = sol.y[0]

    fig, ax = fc.new_figure()
    ax.plot(t, r, label = '$r$')
    ax.plot(t, y, '--', label = '$y$')
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend()
    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/tracking_pendulum3.pdf', pad_inches=0.0)

def main05_robot1():
    """ Second tracking pendulum example """
    x10, x20, y10, y20 = 0, 0, 0, 0
    x0 = np.array([x10, x20, y10, y20])
    tau = 0.25
    omega, zeta = 1/tau, 1
    a, b, c = 10, 1, 10
    k1, k2 = 400, 20

    w = lambda t : 0*t + pi/2
    t_start, t_end = 0, 2
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    umin, umax = -2, 2
    u = lambda t, x, e, r_dd: 1/c * (a*sin(x[0]) + b*x[1] + r_dd - k1*e[0] - k2*e[1])
    u_saturated = lambda t, x, e, r_dd : np.clip(u(t,x,e,r_dd), umin, umax)
    rm_func = lambda t, x, w : reference_model(t, x, w, omega, zeta)
    f = lambda t, x : pendulum_simple_simfunc(t, x, rm_func, w, a, b, c, u_saturated)
    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    # Collect outputs
    w = w(t)
    y1 = sol.y[2]
    y2 = sol.y[3]
    r = y1
    r_dot = y2
    r_dd = -np.power(omega,2)*y1 - 2*zeta*omega*y2 + np.power(omega,2)*w
    x = np.vstack([sol.y[0], sol.y[1]])
    e1 = x[0] - r
    e2 = x[1] - r_dot
    e = np.vstack([e1, e2])
    u = u_saturated(t, x, e, r_dd)
    y = sol.y[0]

    fig, axs = fc.new_figure(subplot_count=2)
    axs[0].plot(t, r, label = '$r$')
    axs[0].plot(t, y, '--', label = '$y$')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(t, u, label = '$u$')
    axs[1].grid()
    axs[1].legend()
    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/tracking_robot1.pdf', pad_inches=0.0)

def main06_robot2():
    """ Second tracking pendulum example """
    x10, x20, y10, y20 = 0, 0, 0, 0
    x0 = np.array([x10, x20, y10, y20])
    tau = 0.05
    omega, zeta = 1/tau, 1
    a, b, c = 10, 1, 10
    k1, k2 = 400, 20

    w = lambda t : 0*t + pi/2
    t_start, t_end = 0, 2
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    umin, umax = -2, 2
    u = lambda t, x, e, r_dd: 1/c * (a*sin(x[0]) + b*x[1] + r_dd - k1*e[0] - k2*e[1])
    u_saturated = lambda t, x, e, r_dd : np.clip(u(t,x,e,r_dd), umin, umax)
    rm_func = lambda t, x, w : reference_model(t, x, w, omega, zeta)
    f = lambda t, x : pendulum_simple_simfunc(t, x, rm_func, w, a, b, c, u_saturated)
    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    # Collect outputs
    w = w(t)
    y1 = sol.y[2]
    y2 = sol.y[3]
    r = y1
    r_dot = y2
    r_dd = -np.power(omega,2)*y1 - 2*zeta*omega*y2 + np.power(omega,2)*w
    x = np.vstack([sol.y[0], sol.y[1]])
    e1 = x[0] - r
    e2 = x[1] - r_dot
    e = np.vstack([e1, e2])
    u = u_saturated(t, x, e, r_dd)
    y = sol.y[0]

    fig, axs = fc.new_figure(subplot_count=2)
    axs[0].plot(t, r, label = '$r$')
    axs[0].plot(t, y, '--', label = '$y$')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(t, u, label = '$u$')
    axs[1].grid()
    axs[1].legend()
    Path('./figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/tracking_robot2.pdf', pad_inches=0.0)