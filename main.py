import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
import scipy.integrate
import pendulum

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

    Tss = 1; # Constant control input
    f = lambda t, x : pend.state_space_constant_control(t, x, Tss)

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    plt.title('Pendulum simulation, $T_{ss} = ' + str(Tss) + '$')
    plt.plot(sol.t, sol.y[0], label = '$\\theta$')
    plt.plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend()
    plt.show()

def main02_constant_input_simulation_fixed_delta():
    """ Simple simulation of a pendulum system. Choose a constant 
        desired output angle delta; a constant control input Tss
        will be calculated to obtain a stable equilibrium at the 
        desired output angle delta. """
    m, g, l, k = 1, 10, 1, 2
    a, b, c = g/l, k/m, 1/m/l/l
    pend = pendulum.Pendulum(m, g, l, k)

    theta0, theta_dot0 = -1, 0
    x0 = np.array([theta0, theta_dot0])

    t_start, t_end = 0, 10
    sample_count = 1000
    t = np.linspace(t_start, t_end, sample_count)
    tspan = [t_start, t_end]

    delta = pi/4
    Tss = a/c * sin(delta);
    f = lambda t, x : pend.state_space_constant_control(t, x, Tss)

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    plt.title('Pendulum simulation, $\\delta = ' + str(delta) + '$')
    plt.plot([t_start, t_end], [pi/4, pi/4], 'r--', label='Reference $\\delta$')
    plt.plot(sol.t, sol.y[0], label = '$\\theta$')
    plt.plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend()
    plt.show()

def main03_constant_input_simulation_shifted():
    """ Simple simulation of a pendulum system. Choose a constant 
        desired output angle delta; a constant control input Tss
        will be calculated to obtain a stable equilibrium at the 
        desired output angle delta. The system's state space
        model is shifted in a way that places the stable 
        equilibrium at the origin. """
    m, g, l, k = 1, 10, 1, 2
    a, b, c = g/l, k/m, 1/m/l/l
    delta = pi/4
    pend = pendulum.Pendulum(m, g, l, k)
    pend.set_equilibrium(delta)

    x1_0, x2_0 = -delta, 0 # Start at theta = zero, theta_dot = 0
    x0 = np.array([x1_0, x2_0])

    t_start, t_end = 0, 10
    N = 1000
    t = np.linspace(t_start, t_end, N)
    tspan = [t_start, t_end]

    uss = 0 # Set torque to constant Tss
    f = lambda t, x : pend.state_space_shifted(t, x, uss)

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    plt.title('Pendulum (equilibrium shifted to origin)')
    plt.plot(sol.t, sol.y[0], label = '$x_1$')
    plt.plot(sol.t, sol.y[1], label = '$x_2$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend()
    plt.show()

def main05_linear_controller():
    """ Use a controller that stabilizes via linearization """
    m, g, l, k = 1, 10, 1, 2
    a, b, c = g/l, k/m, 1/m/l/l
    delta = pi/4
    pend = pendulum.Pendulum(m, g, l, k)

    theta0 , theta_dot0 = -1, 0
    x0 = np.array([theta0, theta_dot0])

    k1, k2 = 1, 1
    u = lambda t, x: -k1*(x[0]-delta) - k2*x[1] + a*sin(delta)/c
    pend.set_control_law(u)

    t_start, t_end = 0, 10
    N = 1000
    t = np.linspace(t_start, t_end, N)
    tspan = [t_start, t_end]

    f = pend.state_space_controlled

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)
    control = u(sol.t, sol.y)

    plt.title('Pendulum (stabilization by linearization)')
    plt.plot([t_start, t_end], [pi/4, pi/4], 'r--', label='Reference $\\delta$')
    plt.plot(sol.t, sol.y[0], label = '$\\theta$')
    plt.plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    plt.plot(sol.t, control, label = '$T$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend()
    plt.show()

def helper_func(t, x):
    m, g, l, k = 1, 10, 1, 2
    a, b, c = g/l, k/m, 1/m/l/l
    delta = pi/4
    A = np.array([[0, 1], [-a*cos(delta), -b]])
    B = np.array([[0],[c]])
    C = np.array([[1, 0]])

    k1, k2, h1, h2 = 1, 1, 1, 1
    K = np.array([[k1, k2]])
    H = np.array([[h1], [h2]])

    x1 = x[0]
    x2 = x[1]
    x1_hat = x[2]
    x2_hat = x[3]
    x = np.array([[x1], [x2]])
    x_hat = np.array([[x1_hat], [x2_hat]])

    x1_dot = x2
    x2_dot = -a*(sin(x1+delta)-sin(delta)) - b*x2 - c*K@x_hat
    xhat_dot = H@C@x + (A-B@K-H@C)@x_hat

    x1_hat_dot = xhat_dot[0,0]
    x2_hat_dot = xhat_dot[1,0]
    x2_dot = x2_dot[0,0]
    x_total_dot = np.array([x1_dot, x2_dot, x1_hat_dot, x2_hat_dot])
    return x_total_dot

def main06_output_feedback_stabilization():
    m, g, l, k = 1, 10, 1, 2
    a, b, c = g/l, k/m, 1/m/l/l
    delta = pi/4
    pend = pendulum.Pendulum(m, g, l, k)
    pend.set_equilibrium(delta)

    x0 = np.array([-delta, 0, -delta, 0])
    t_start, t_end = 0, 10
    N = 1000
    t = np.linspace(t_start, t_end, N)
    tspan = [t_start, t_end]

    f = helper_func

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    plt.title('Pendulum (stabilization by linearization, output feedback)')
    plt.plot(sol.t, sol.y[0], label = '$x_1$')
    plt.plot(sol.t, sol.y[1], label = '$x_2$')
    plt.plot(sol.t, sol.y[2], label = '$\\hat{x_1}$')
    plt.plot(sol.t, sol.y[3], label = '$\\hat{x_2}$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend()
    plt.show()

def main07_output_feedback_stabilization():
    m, g, l, k = 1, 10, 1, 2
    delta = pi/4
    pend = pendulum.Pendulum(m, g, l, k)
    pend.set_equilibrium(delta)

    x0 = np.array([0, 0, 0, 0])
    t_start, t_end = 0, 10
    N = 1000
    t = np.linspace(t_start, t_end, N)
    tspan = [t_start, t_end]

    f = pend.state_space_output_feedback

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    plt.title('Pendulum (stabilization by linearization, output feedback)')
    plt.plot([t_start, t_end], [delta, delta], 'r--', label='Reference $\\delta$')
    plt.plot(sol.t, sol.y[0], label = '$\\theta$')
    plt.plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    plt.plot(sol.t, sol.y[2], label = '$\\hat{\\theta}$')
    plt.plot(sol.t, sol.y[3], label = '$\\hat{\\dot{\\theta}}$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend()
    plt.show()

def helper_function2(t, alpha):
    m, g, l, k = 1, 10, 1, 2
    a, b, c = g/l, k/m/2, 1/m/l/l/2
    delta = pi/4

    k1, k2, k3 = 8, 2, 10
    K = np.array([[k1, k2, k3]])

    # Handle inputs
    x1 = alpha[0] - delta
    x2 = alpha[1]
    sigma = alpha[2]

    # Calculate control
    u = - K @ np.array([[x1],[x2],[sigma]])
    u = u[0,0]

    # Calculate state derivatives
    x1_dot = x2
    x2_dot = -a*sin(x1+delta) - b*x2 + c*u
    sigma_dot = x1

    alpha_dot = np.array([x1_dot, x2_dot, sigma_dot])
    return alpha_dot

def main10_integral_control():
    m, g, l, k = 1, 10, 1, 2
    delta = pi/4
    k1, k2, k3 = 10, 1, 1
    K = np.array([[k1, k2, k3]])

    x0 = np.array([0, 0, 0])
    t_start, t_end = 0, 10
    N = 1000
    t = np.linspace(t_start, t_end, N)
    tspan = [t_start, t_end]

    f = helper_function2

    sol = scipy.integrate.solve_ivp(f, tspan, x0, t_eval=t)

    plt.title('Pendulum (stabilization by linearization, output feedback)')
    plt.plot([t_start, t_end], [delta, delta], 'r--', label='Reference $\\delta$')
    plt.plot(sol.t, sol.y[0], label = '$\\theta$')
    plt.plot(sol.t, sol.y[1], label = '$\\dot{\\theta}$')
    plt.plot(sol.t, sol.y[2], label = '$\\sigma$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend()
    plt.show()

def main11_test_integral_control_parameters():
    m, g, l, k = 1, 10, 1, 2
    a, b, c = g/l, k/m, 1/m/l/l
    k1, k2, k3 = 8, 2, 10
    delta = pi/4

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

if __name__ == '__main__':
    print('Program start!')

    # main01_constant_input_simulation()
    main02_constant_input_simulation_fixed_delta()
    # main03_keep_steady_state()
    # main04_keep_steady_state()
    # main05_linear_controller()
    # main06_output_feedback_stabilization()
    # main07_output_feedback_stabilization()
    # main10_integral_control()
    # main11_test_integral_control_parameters()

    print('Program end!')
