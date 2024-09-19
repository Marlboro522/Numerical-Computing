import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return -12 - 21*x + 18*x**2 - 2.75*x**3

# Define the range of x values
x_values = np.linspace(-10, 10, num=100)
y_values = f(x_values)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='f(x) = -12 - 21x + 18x^2 - 2.75x^3')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = -12 - 21x + 18x^2 - 2.75x^3')
plt.legend()
plt.grid(True)
plt.show()

"""Implementations of the roots Bracketing methods here. """
def bisection_method(func, a, b, tol=0.1, i_max=20):
    if func(a) * func(b) >= 0:
        print("Doesn't work f(a) * f(b) should be less than 0")
        return None
    i_c = 0
    while (b - a) / 2 > tol and i_c < i_max:
        midpoint = (a + b) / 2
        if func(midpoint) == 0:
            return midpoint
        elif func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        i_c += 1
    return (a + b) / 2

root_bisection = bisection_method(f, -10, 0)
print(f"Root found by Bisection Method: {root_bisection}")

def false_position_method(func, a, b, tol=0.1, max_iter=20):
    if func(a) * func(b) >= 0:
        print("Doesn't work f(a) * func(b) less than 0")
        return None
    iter_count = 0
    while abs(b - a) > tol and iter_count < max_iter:
        root = b - (func(b) * (b - a)) / (func(b) - func(a))
        if func(root) == 0:
            return root
        elif func(a) * func(root) < 0:
            b = root
        else:
            a = root
        iter_count += 1
    return root

root_false_position = false_position_method(f, -10, 0)
print(f"Root found by False Position Method: {root_false_position}")

""" Implementation for the Roots open methods """

def g(x):
    return x**3 - 6*x**2 + 11*x - 6.1

# Define the range of x values
x_values = np.linspace(0, 4, 400)
y_values = g(x_values)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='f(x) = x^3 - 6x^2 + 11x - 6.1')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = x^3 - 6x^2 + 11x - 6.1')
plt.legend()
plt.grid(True)
plt.show()

""" Newton Raphson Method """

def newton_raphson(func, dfunc, x0, tol=1e-6, max_iter=3):
    x = x0
    for _ in range(max_iter):
        x_new = x - func(x) / dfunc(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
        print(x_new)
    return x

def dg(x):
    return 3*x**2 - 12*x + 11

root_newton = newton_raphson(g, dg, 3.5)
print(f"Root found by Newton-Raphson Method for problem 2: {root_newton}")
    
def h(x):
    return 0.0074*x**4 - 0.284*x**3 + 3.355*x**2 - 12.183*x + 5

""" Function h(x) differntiated"""
def dh(x):
    return 4*0.0074*x**3 - 3*0.284*x**2 + 2*3.355*x - 12.183

root_newton_h = newton_raphson(h, dh, 16.15)
print(f"Root found by Newton-Raphson Method: {root_newton_h}")


