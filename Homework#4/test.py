from scipy.integrate import quad

# Define the function
def f(x):
    return -0.055*x**4 + 0.86*x**3 - 4.2*x**2 + 6.3*x + 2

# Compute the integral
result, error = quad(f, 0, 8)

print(f"Integral: {result}, Estimated Error: {error}")


import numpy as np

def finite_difference(x, y):
    """
    Computes the first and second derivatives.

    Parameters:
        x : x values
        y : y values

    Returns:
        d1 : First derivatives at eac point
        d2 : Second derivatives at eachh point
    """
    h = x[1] - x[0]  # Assuming equally spaced points

    # Compute finite differences
    dy = np.diff(y)  # First differences
    ddy = np.diff(dy)  # Second differences

    # First Derivative
    d1 = np.zeros(len(x))
    d1[1:-1] = dy[1:] / h  # Central difference using dy
    d1[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * h)  # Forward difference
    d1[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * h)  # Backward difference

    # Second Derivative
    d2 = np.zeros(len(x))
    d2[1:-1] = ddy / (h ** 2)  # Central difference using ddy
    d2[0] = (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / (h ** 2)  # Forward difference
    d2[-1] = (2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]) / (h ** 2)  # Backward difference

    return d1, d2

# Data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1.4, 2.1, 3.3, 4.8, 6.8, 6.6, 8.6, 7.5, 8.9, 10.9])

# Compute derivatives
d1, d2 = finite_difference(x, y)

# Print results
print("First Derivatives:")
print(d1)

print("\nSecond Derivatives:")
print(d2)
