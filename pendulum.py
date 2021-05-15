import numpy as np
from numpy import sin, cos

class Pendulum:
    def __init__(self, m, g, l, k):
        self.m = m
        self.g = g
        self.l = l
        self.k = k

        self.a = g/l
        self.b = k/m
        self.c = 1/m/l/l

    def set_control_law(self, control_func):
        """ Control law should have the form u = u(t, x) """
        self.u = control_func

    def state_space_constant_control(self, t, x, T):
        x1 = x[0]
        x2 = x[1]

        x1_dot = x2
        x2_dot = -self.a*sin(x1) - self.b*x2 + self.c*T

        x_dot = np.array([x1_dot, x2_dot])
        return x_dot

    def state_space_controlled(self, t, x):
        x1 = x[0]
        x2 = x[1]

        T = self.u(t, x)
        x1_dot = x2
        x2_dot = -self.a*sin(x1) - self.b*x2 + self.c*T

        x_dot = np.array([x1_dot, x2_dot])
        return x_dot

    def state_space_shifted(self, t, x, u):
        x1 = x[0] # x1 = theta - delta
        x2 = x[1] # x2 = theta_dot

        x1_dot = x2
        x2_dot = -self.a*(sin(x1+self.delta) - sin(self.delta)) - self.b*x2 + self.c*u

        x_dot = np.array([x1_dot, x2_dot])
        return x_dot

    def state_space_output_feedback(self, t, z):
        k1, k2, h1, h2 = 1, 1, -self.b, 1
        K = np.array([[k1, k2]])
        H = np.array([[h1], [h2]])

        A = np.array([[0, 1], [-self.a*cos(self.delta), -self.b]])
        B = np.array([[0],[self.c]])
        C = np.array([[1, 0]])

        # Change variables so equilibrium is at origin
        z_shifted = z - np.array([self.delta, 0, self.delta, 0])
        x_shifted = z_shifted[0:2]
        x_hat_shifted = z_shifted[2:4]

        # Calculate control based on linearized model and fetch x_dot
        u = -np.matmul(K, x_hat_shifted)[0]
        x_dot = self.state_space_shifted(t, x_shifted, u)

        # Calculate observer dynamics
        x_shifted_m = x_shifted[np.newaxis].T
        x_hat_shifted_m = x_hat_shifted[np.newaxis].T
        x_hat_dot = H@C@x_shifted_m + (A-B@K-H@C)@x_hat_shifted_m
        x_hat_dot = x_hat_dot.ravel()

        z_shifted_dot = np.concatenate([x_dot, x_hat_dot])
        return z_shifted_dot;

    def state_space_integral_control(self, t, alpha):
        k1, k2, k3 = 1, 1, 1
        K = np.array([[k1, k2, k3]])

        A = np.array([[0, 1], [-self.a*cos(self.delta), -self.b]])
        B = np.array([[0],[self.c]])
        C = np.array([[1, 0]])
