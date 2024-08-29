import numpy as np
from scipy.optimize import minimize_scalar

v0 = 15
h1 = 0.6
h2 = 10
L = 0.4
g = 9.81

def distance_x1(theta):
    theta_rad = np.radians(theta)
    a = 0.5 * g
    b = -v0 * np.sin(theta_rad)
    c = h2 - h1
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        return float('inf')
    
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)
    
    t = max(t1, t2)
    
    if t <= 0:
        return float('inf')
    
    x1 = v0 * np.cos(theta_rad) * t
    return x1

def coverage_distance(theta):
    x1 = distance_x1(theta)
    if x1 == float('inf'):
        return float('inf')
    x2 = x1 + L
    return -(x2 - x1)

result = minimize_scalar(coverage_distance, bounds=(0, 90), method='bounded')

if result.success:
    optimal_theta = result.x
    optimal_x1 = distance_x1(optimal_theta)
    optimal_coverage = -result.fun
    
    print(f"Optimal angle (theta): {optimal_theta:.2f} degrees")
    print(f"Optimal distance from the building (x1): {optimal_x1:.2f} meters")
    print(f"Maximum coverage distance (x2 - x1): {optimal_coverage:.2f} meters")
else:
    print("Optimization did not converge.")
