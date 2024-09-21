import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

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
plt.savefig('graph1.png')

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
plt.savefig("graph_2.png")

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


"""Problem 4"""

def f_opt(x):
    return 4*x - 1.8*x**2 + 1.2*x**3 - 0.3*x**4

# Define the range of x values
x_values = np.linspace(-1, 3, 400)
y_values = f_opt(x_values)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='f(x) = 4x - 1.8x^2 + 1.2x^3 - 0.3x^4')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = 4x - 1.8x^2 + 1.2x^3 - 0.3x^4')
plt.legend()
plt.grid(True)
plt.savefig("grapj_3.png")

def df_opt(x):
    return 4 - 3.6*x + 3.6*x**2 - 1.2*x**3

#for findign the toors of the fucntioon
roots_opt = fsolve(df_opt, [0, 1, 2])
print(f"Critical points found: {roots_opt}")


"""Golden SEction Search"""

def golden_section_search(func, a, b, tol=1e-2):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = func(x1)
    f2 = func(x2)
    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a + resphi * (b - a)
            f2 = f1
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            x2 = b - resphi * (b - a)
            f1 = f2
            f2 = func(x2)
    return (a + b) / 2

x_golden = golden_section_search(f_opt, 0, 2)
print(f"Maximum found by Golden Section Search: {x_golden}")


"""Parabolic Interpolation"""

def parabolic_interpolation(func, x1, x2, x3, tol=1e-2):
    def fit_parabola(x1, x2, x3):
        f1, f2, f3 = func(x1), func(x2), func(x3)
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        A = (x3 * (f2 - f1) + x2 * (f1 - f3) + x1 * (f3 - f2)) / denom
        B = (x3**2 * (f1 - f2) + x2**2 * (f3 - f1) + x1**2 * (f2 - f3)) / denom
        C = (x2 * x3 * (x2 - x3) * f1 + x3 * x1 * (x3 - x1) * f2 + x1 * x2 * (x1 - x2) * f3) / denom
        return A, B, C

    iter_count = 0
    while abs(x3 - x1) > tol and iter_count < 100:
        A, B, C = fit_parabola(x1, x2, x3)
        x_new = -B / (2 * A)
        if x_new < x2:
            if func(x_new) < func(x2):
                x3 = x2
                x2 = x_new
            else:
                x1 = x_new
        else:
            if func(x_new) < func(x2):
                x1 = x2
                x2 = x_new
            else:
                x3 = x_new
        iter_count += 1
    return x2

x_parabolic = parabolic_interpolation(f_opt, 0, 1, 2)
print(f"Maximum found by Parabolic Interpolation: {x_parabolic}")

"""Hose Problem"""
# import numpy as np
# import scipy.optimize as opt

# # Given values
# v = 15  
# g = 9.81  
# h1 = 0.6 
# h2 = 10 
# L = 0.4 

# #maximizing the coverage/
# def coverage(params):
#     x1, theta = params
#     theta_rad = np.radians(theta)
#     t = (L - x1) / (v * np.cos(theta_rad))
#     y_t = h1 + v * np.sin(theta_rad) * t - 0.5 * g * t**2
#     return -(y_t - h2)  # Negative because we use minimize function

# #this can be the constraint
# def constraint(params):
#     x1, theta = params
#     theta_rad = np.radians(theta)
#     t = (L - x1) / (v * np.cos(theta_rad))
#     y_t = h1 + v * np.sin(theta_rad) * t - 0.5 * g * t**2
#     return y_t - h2

# # Initial guesses
# initial_guess = [0.1, 45]  # x1 = 0.1 m, theta = 45 degrees

# # Bounds for x1 and theta
# bounds = [(0, L), (0, 90)]

# # Optimization using SLSQP
# result = opt.minimize(coverage, initial_guess, bounds=bounds, constraints={'type': 'eq', 'fun': constraint}, method='SLSQP')

# # Results
# x1_opt, theta_opt = result.x
# x2_opt = L
# coverage_opt = x2_opt - x1_opt

# print(f"Optimal x1: {x1_opt:.4f} m")
# print(f"Optimal θ: {theta_opt:.4f} degrees")
# print(f"Maximum coverage: {coverage_opt:.4f} m")

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Given constants
v = 15  # m/s
g = 9.81  # m/s^2
h1 = 0.6  # m
h2 = 10  # m
L = 0.4  # m

# Define the objective function with penalty for constraint violation
def coverage_with_penalty(params):
    x1, theta = params
    theta_rad = np.radians(theta)
    t = (L - x1) / (v * np.cos(theta_rad))
    y_t = h1 + v * np.sin(theta_rad) * t - 0.5 * g * t**2
    penalty = 0
    if y_t < h2:  # Constraint violation penalty
        penalty = 1e6 * (h2 - y_t)**2  # Penalty term
    if not (0 <= x1 <= L):  # x1 out of bounds penalty
        penalty += 1e6 * (min(0, x1) ** 2 + max(0, x1 - L) ** 2)
    if not (0 <= theta <= 90):  # theta out of bounds penalty
        penalty += 1e6 * (min(0, theta) ** 2 + max(0, theta - 90) ** 2)
    return -(y_t - h2) + penalty  # Negative because we use minimize function

# Initial guesses
initial_guess = [0.1, 45]  # x1 = 0.1 m, theta = 45 degrees

# Bounds for x1 and theta (note: Nelder-Mead does not support bounds natively, but we can manually enforce them)
bounds = [(0, L), (0, 90)]

# Custom function to enforce bounds manually
def enforce_bounds(params):
    x1, theta = params
    x1 = np.clip(x1, bounds[0][0], bounds[0][1])
    theta = np.clip(theta, bounds[1][0], bounds[1][1])
    return [x1, theta]

# Optimization using Nelder-Mead (Simplex) method
result = opt.minimize(lambda params: coverage_with_penalty(enforce_bounds(params)), initial_guess, method='Nelder-Mead')

# Enforce bounds on the result
x1_opt = np.clip(result.x[0], 0, L)
theta_opt = np.clip(result.x[1], 0, 90)
x2_opt = L
coverage_opt = x2_opt - x1_opt

print(f"Optimal x1: {x1_opt:.4f} m")
print(f"Optimal θ: {theta_opt:.4f} degrees")
print(f"Maximum coverage: {coverage_opt:.4f} m")

#Not really sure abot the results, a simmulationn woul dbe useful.