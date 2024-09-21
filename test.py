# # # # def f1(a):
# # # #     # print(a **3 - 6 * a **2 + 11 * a - 6.1)
# # # #     return a **3 - 6 * a **2 + 11 * a - 6.1
# # # # # def f2(a):
# # # # #     print(3 * a **2 - 12 * a + 11)
# # # # #     return 3 * a **2 - 12 * a + 11
# # # # def result(ini):
# # # #     return ini-((f1(ini) * (0.01 * ini)) / ((f1(ini + 0.01 *ini )) - f1(ini)))
# # # # print(result(3.0467729))

# # # import numpy as np
# # # import scipy.optimize as opt
# # # import matplotlib.pyplot as plt

# # # # Given constants
# # # v = 15  # m/s, velocity of water leaving the hose
# # # g = 9.81  # m/s^2, acceleration due to gravity
# # # h1 = 0.6  # m, initial height of the water stream from the ground
# # # h2 = 10  # m, height of the roof edge
# # # L = 0.4  # m, horizontal distance to the roof edge

# # # # Define the objective function with penalty for constraint violation
# # # def coverage_with_penalty(params):
# # #     x1, theta = params
# # #     theta_rad = np.radians(theta)
# # #     t = (L + x1) / (v * np.cos(theta_rad))  # Adjusted to include x1 as distance from building
# # #     y_t = h1 + v * np.sin(theta_rad) * t - 0.5 * g * t**2
# # #     penalty = 0
# # #     if y_t < h2:  # Constraint violation penalty
# # #         penalty = 1e8 * (h2 - y_t)**2  # Increased Penalty term
# # #     if not (0 <= x1 <= 10):  # x1 out of bounds penalty, assuming reasonable range
# # #         penalty += 1e8 * (min(0, x1) ** 2 + max(0, x1 - 10) ** 2)
# # #     if not (0 <= theta <= 90):  # theta out of bounds penalty
# # #         penalty += 1e8 * (min(0, theta) ** 2 + max(0, theta - 90) ** 2)
# # #     return -(y_t - h2) + penalty  # Negative because we use minimize function

# # # # Initial guesses
# # # initial_guesses = [
# # #     [4, 45],  # x1 = 4 m, theta = 45 degrees
# # #     [4, 30],  # x1 = 4 m, theta = 30 degrees
# # #     [4, 60]   # x1 = 4 m, theta = 60 degrees
# # # ]

# # # # Custom function to enforce bounds manually
# # # def enforce_bounds(params):
# # #     x1, theta = params
# # #     x1 = np.clip(x1, 0, 10)
# # #     theta = np.clip(theta, 0, 90)
# # #     return [x1, theta]

# # # best_result = None

# # # # Run optimization for each initial guess
# # # for initial_guess in initial_guesses:
# # #     result = opt.minimize(lambda params: coverage_with_penalty(enforce_bounds(params)), initial_guess, method='Nelder-Mead')
# # #     if best_result is None or result.fun < best_result.fun:
# # #         best_result = result

# # # # Enforce bounds on the result
# # # x1_opt = np.clip(best_result.x[0], 0, 10)
# # # theta_opt = np.clip(best_result.x[1], 0, 90)
# # # x2_opt = L + x1_opt  # Adjusted to include the distance from the building
# # # coverage_opt = x2_opt - x1_opt

# # # print(f"Optimal x1: {x1_opt:.4f} m")
# # # print(f"Optimal θ: {theta_opt:.4f} degrees")
# # # print(f"Maximum coverage: {coverage_opt:.4f} m")
# # def f(x,y):
# #     return (6 * (x ** 2))-(2*y)+1
# # print(f(2,2))


import numpy as np
import scipy.optimize as opt

v = 15  
g = 9.81 
h1 = 0.6 
h2 = 10  
L = 0.4  

def y_position(x1, theta):
    theta_rad = np.radians(theta)
    x_total = x1 - L  # since L is where the stream starts. 
    return h1 + x_total * np.tan(theta_rad) - (g * x_total**2) / (2 * v**2 * np.cos(theta_rad)**2)

def constraint(params):
    x1, theta = params
    return y_position(x1, theta) - h2  #since the height has to exceed 10m. 

def coverage_objective(params):
    x1, theta = params
    theta_rad = np.radians(theta)
    t_flight = (v * np.sin(theta_rad) + np.sqrt((v * np.sin(theta_rad))**2 + 2 * g * h1)) / g
    x2 = x1 + v * np.cos(theta_rad) * t_flight
    print(-(x2-x1))
    return -(x2 - x1)


initial_guess = [5, 15] 
bounds = [(0, 20), (0, 90)] 
constraints = {'type': 'ineq', 'fun': constraint}  #ineq beacuse we can ensure that the projection is greater than 10m whcih is height of the wall

result = opt.minimize(coverage_objective, initial_guess, method='SLSQP', constraints=constraints, bounds=bounds)

x1_opt = result.x[0]
theta_opt = result.x[1]


print(result.message)
print(f"Optimal x1: {x1_opt:.4f} m")
print(f"Optimal θ: {theta_opt:.4f} degrees")

# import matplotlib.pyplot as plt
# angles = np.linspace(0, 90, 100)
# trajectories = []

# for angle in angles:
#     y_vals = [y_position(x1, angle) for x1 in np.linspace(0, 20, 100)]
#     trajectories.append(y_vals)

# # Plot
# for trajectory in trajectories:
#     plt.plot(np.linspace(0, 20, 100), trajectory)

# plt.axhline(y=h2, color='r', linestyle='--', label='Height of roof (10m)')
# plt.xlabel('Horizontal Distance (m)')
# plt.ylabel('Height (m)')
# plt.title('Projectile Trajectories')
# plt.legend()
# plt.show()