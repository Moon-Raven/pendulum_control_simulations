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

    def state_space_static_controller(self, t, x, T):
        """ 
            T (control law) should have the form T = T(t, x), where:
            x1 = theta
            x2 = theta_dot
        """
        x1 = x[0]
        x2 = x[1]

        x1_dot = x2
        x2_dot = -self.a*sin(x1) - self.b*x2 + self.c*T(t, x)

        x_dot = np.array([x1_dot, x2_dot])
        return x_dot

    def state_space_observer_controller(self, t, alpha, K, H, delta):
        A = np.array([[0, 1], [-self.a*cos(delta), -self.b]])
        B = np.array([[0],[self.c]])
        C = np.array([[1, 0]])

        # Change variables so equilibrium is at origin
        variable_offset = np.array([delta, 0])
        x = alpha[0:2]
        x_shifted = x - variable_offset
        x_hat_shifted = alpha[2:4] - variable_offset

        # Calculate control based on linearized model and fetch x_dot
        T_shifted = -np.dot(K, x_hat_shifted)
        Tss = self.a/self.c * sin(delta)
        T_func = lambda t, x : Tss + T_shifted
        x_dot = self.state_space_static_controller(t, x, T_func)

        # Calculate observer dynamics
        K = K[np.newaxis]   # Convert from np array to np matrix
        H = H[np.newaxis].T # Convert from np array to np matrix
        x_shifted_m = x_shifted[np.newaxis].T
        x_hat_shifted_m = x_hat_shifted[np.newaxis].T
        x_hat_dot = H@C@x_shifted_m + (A-B@K-H@C)@x_hat_shifted_m
        x_hat_dot = x_hat_dot.ravel()

        # Bear in mind that alpha_dot = alpha_shifted dot, becuase derivative
        # of a constant iz zero
        alpha_dot = np.concatenate([x_dot, x_hat_dot])
        return alpha_dot

    def state_space_integral_controller(self, t, alpha, K, delta):
        # Handle inputs
        variable_offset = np.array([delta, 0, 0])
        alpha_shifted = alpha - variable_offset
        x1 = alpha[0]
        x2 = alpha[1]
        x1_shifted = alpha_shifted[0]
        x2_shifted = alpha_shifted[1]
        sigma = alpha_shifted[2]

        # Calculate control
        T_shifted = -np.dot(K, alpha_shifted)

        # Calculate state derivatives
        x1_dot = x2
        x2_dot = -self.a*sin(x1) - self.b*x2 + self.c*T_shifted
        sigma_dot = x1_shifted

        alpha_dot = np.array([x1_dot, x2_dot, sigma_dot])

        return alpha_dot