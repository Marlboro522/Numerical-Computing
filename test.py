# # # def f1(a):
# # #     # print(a **3 - 6 * a **2 + 11 * a - 6.1)
# # #     return a **3 - 6 * a **2 + 11 * a - 6.1
# # # # def f2(a):
# # # #     print(3 * a **2 - 12 * a + 11)
# # # #     return 3 * a **2 - 12 * a + 11
# # # def result(ini):
# # #     return ini-((f1(ini) * (0.01 * ini)) / ((f1(ini + 0.01 *ini )) - f1(ini)))
# # # print(result(3.0467729))

# # import numpy as np
# # import scipy.optimize as opt
# # import matplotlib.pyplot as plt

# # # Given constants
# # v = 15  # m/s, velocity of water leaving the hose
# # g = 9.81  # m/s^2, acceleration due to gravity
# # h1 = 0.6  # m, initial height of the water stream from the ground
# # h2 = 10  # m, height of the roof edge
# # L = 0.4  # m, horizontal distance to the roof edge

# # # Define the objective function with penalty for constraint violation
# # def coverage_with_penalty(params):
# #     x1, theta = params
# #     theta_rad = np.radians(theta)
# #     t = (L + x1) / (v * np.cos(theta_rad))  # Adjusted to include x1 as distance from building
# #     y_t = h1 + v * np.sin(theta_rad) * t - 0.5 * g * t**2
# #     penalty = 0
# #     if y_t < h2:  # Constraint violation penalty
# #         penalty = 1e8 * (h2 - y_t)**2  # Increased Penalty term
# #     if not (0 <= x1 <= 10):  # x1 out of bounds penalty, assuming reasonable range
# #         penalty += 1e8 * (min(0, x1) ** 2 + max(0, x1 - 10) ** 2)
# #     if not (0 <= theta <= 90):  # theta out of bounds penalty
# #         penalty += 1e8 * (min(0, theta) ** 2 + max(0, theta - 90) ** 2)
# #     return -(y_t - h2) + penalty  # Negative because we use minimize function

# # # Initial guesses
# # initial_guesses = [
# #     [4, 45],  # x1 = 4 m, theta = 45 degrees
# #     [4, 30],  # x1 = 4 m, theta = 30 degrees
# #     [4, 60]   # x1 = 4 m, theta = 60 degrees
# # ]

# # # Custom function to enforce bounds manually
# # def enforce_bounds(params):
# #     x1, theta = params
# #     x1 = np.clip(x1, 0, 10)
# #     theta = np.clip(theta, 0, 90)
# #     return [x1, theta]

# # best_result = None

# # # Run optimization for each initial guess
# # for initial_guess in initial_guesses:
# #     result = opt.minimize(lambda params: coverage_with_penalty(enforce_bounds(params)), initial_guess, method='Nelder-Mead')
# #     if best_result is None or result.fun < best_result.fun:
# #         best_result = result

# # # Enforce bounds on the result
# # x1_opt = np.clip(best_result.x[0], 0, 10)
# # theta_opt = np.clip(best_result.x[1], 0, 90)
# # x2_opt = L + x1_opt  # Adjusted to include the distance from the building
# # coverage_opt = x2_opt - x1_opt

# # print(f"Optimal x1: {x1_opt:.4f} m")
# # print(f"Optimal θ: {theta_opt:.4f} degrees")
# # print(f"Maximum coverage: {coverage_opt:.4f} m")
# def f(x,y):
#     return (6 * (x ** 2))-(2*y)+1
# print(f(2,2))
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Given constants
v = 15  # m/s, velocity of water leaving the hose
g = 9.81  # m/s^2, acceleration due to gravity
h1 = 0.6  # m, initial height of the water stream from the ground
h2 = 10  # m, height of the roof edge
L = 0.4  # m, length of the hose

# Define the equation for the vertical position y
def y_position(x1, theta):
    theta_rad = np.radians(theta)
    x_total = x1 + L
    return h1 + x_total * np.tan(theta_rad) - (g * x_total**2) / (2 * v**2 * np.cos(theta_rad)**2)

# Constraint function to ensure the stream clears the roof edge
def constraint(params):
    x1, theta = params
    return y_position(x1, theta) - h2

# Define the objective function to maximize the horizontal distance covered
def coverage_objective(params):
    x1, theta = params
    theta_rad = np.radians(theta)
    t_flight = (v * np.sin(theta_rad) + np.sqrt((v * np.sin(theta_rad))**2 + 2 * g * h1)) / g
    x2 = x1 + v * np.cos(theta_rad) * t_flight
    return -(x2 - x1)  # We want to maximize the horizontal distance (negative for minimization)

# Initial guesses
initial_guess = [4, 45]  # x1 = 4 m, theta = 45 degrees

# Bounds for the variables
bounds = [(0, 20), (0, 90)]  # x1 can be from 0 to 20 meters, theta can be from 0 to 90 degrees

# Define constraints dictionary
constraints = {'type': 'ineq', 'fun': constraint}

# Optimization using SLSQP method
result = opt.minimize(coverage_objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal values
x1_opt = result.x[0]
theta_opt = result.x[1]

print(f"Optimal x1: {x1_opt:.4f} m")
print(f"Optimal θ: {theta_opt:.4f} degrees")

# Function to calculate the trajectory of the water stream
def trajectory(x1, theta, num_points=500):
    theta_rad = np.radians(theta)
    t_flight = (v * np.sin(theta_rad) + np.sqrt((v * np.sin(theta_rad))**2 + 2 * g * h1)) / g
    t = np.linspace(0, t_flight, num_points)
    x = x1 + v * np.cos(theta_rad) * t
    y = h1 + v * np.sin(theta_rad) * t - 0.5 * g * t**2
    return x, y

# Simulate the trajectory with optimal parameters
x_traj, y_traj = trajectory(x1_opt, theta_opt)

# Plotting the trajectory
plt.figure(figsize=(12, 8))
plt.plot(x_traj, y_traj, label=f'Trajectory at θ={theta_opt:.2f}°, x1={x1_opt:.2f} m from building')
plt.axhline(h2, color='red', linestyle='--', label='Roof Height')
plt.axvline(x1_opt, color='green', linestyle='--', label='Front Edge of Roof')
plt.scatter(x1_opt, h1, color='blue', label='Hose Position')
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Height (m)')
plt.title('Water Stream Trajectory')
plt.legend()
plt.grid(True)
plt.show()
