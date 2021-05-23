import numpy as np
from scipy.integrate import trapz

class TimeseriesIntegrator:
    """ Integrator designed to be used in odeint solver functions """
    def __init__(self):
        N = 100000
        self.t = -np.ones(N) * np.nan
        self.y = -np.ones(N) * np.nan
        self.last_index = -1

    def integrate(self, span):
        if self.last_index == -1:
            raise RuntimeError('Integration called, but sample database empty.')

        integration_t, integration_y = self.get_integration_arrays(span)
        res = trapz(integration_y, integration_t)
        return res

    def add_sample(self, t, y):
        larger = self.t[0:self.last_index+1] > t
        if not larger.any():
            self.last_index += 1
        else:
            first_larger = np.argmax(larger)
            self.t[first_larger:self.last_index+1] = -np.nan
            self.y[first_larger:self.last_index+1] = -np.nan
            self.last_index = first_larger
        self.t[self.last_index] = t
        self.y[self.last_index] = y

    def get_y(self, t):
        y = np.interp(t, self.t, self.y)
        return y

    def get_integration_arrays(self, span):
        tleft, tright = span[0], span[1]
        inner_left_index = self._get_inner_left_index(tleft)
        inner_right_index = self._get_inner_right_index(tright)

        inner_size = inner_right_index - inner_left_index + 1
        total_size = inner_size + 2

        integration_t = np.ones(total_size) * np.nan
        integration_y = np.ones(total_size) * np.nan

        # Fill inner arrays
        integration_t[1:-1] = self.t[inner_left_index:inner_right_index+1]
        integration_y[1:-1] = self.y[inner_left_index:inner_right_index+1]

        # Fill edges
        integration_t[0] = tleft
        integration_t[-1] = tright
        integration_y[0] = np.interp(tleft, self.t[0:self.last_index+1], self.y[0:self.last_index+1])
        integration_y[-1] = np.interp(tright, self.t[0:self.last_index+1], self.y[0:self.last_index+1])
        return integration_t, integration_y

    def _get_inner_left_index(self, t):
        are_right = self.t >= t
        return np.argmax(are_right)

    def _get_inner_right_index(self, t):
        are_right = self.t >= t
        if not np.any(are_right):
            inner_right_index = self.last_index
        else:
            inner_right_index = np.argmax(are_right) - 1
            if inner_right_index < 0:
                inner_right_index = 0
        return inner_right_index

def run_tests():
    import pdb; pdb.set_trace()
    tsi = TimeseriesIntegrator()

    # Test case A (zero samples)
    try:
        res = tsi.integrate([1, 3])
    except RuntimeError:
        pass
    else:
        print("Test case A: Exception not thrown")

    # Test case B (one sample)
    tsi.add_sample(1, 0)
    res = tsi.integrate([-4, 6])
    assert res == 0

    tsi.add_sample(2, 2)
    tsi.add_sample(4, 1)

    # Remaining test cases
    eps = 0.01
    print(f'D: {(tsi.integrate([2,3]) - 1.75)}')
    print(f'E: {(tsi.integrate([2, 4]) - 3)}')
    print(f'F: {(tsi.integrate([1, 3]) - 2.75)}')
    print(f'G: {(tsi.integrate([2.5, 3.5]) - 1.5)}')
    print(f'H: {(tsi.integrate([1.5, 3.5]) - 3.1875)}')
    print(f'I: {(tsi.integrate([0, 1.5]) - 0.25)}')
    print(f'J: {(tsi.integrate([3.5, 10]) - 6.5625)}')

    tsi.add_sample(1.5, 4)
    print(f'K: {(tsi.integrate([0, 2]) - 3)}')

    tsi.add_sample(-1, 5)
    print(f'L: {(tsi.integrate([0, 2]) - 10)}')

if __name__ == '__main__':
    run_tests()