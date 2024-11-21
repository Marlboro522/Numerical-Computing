import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the data
data = np.array([
    29.65, 30.65, 29.65, 31.25, 28.55, 28.15, 30.45, 29.45,
    28.65, 29.85, 29.15, 30.15, 30.15, 29.35, 29.05, 30.25,
    30.45, 34.65, 29.65, 30.55, 29.75, 30.85, 29.35, 29.65,
    29.25, 28.75, 29.75, 29.25
])

# Calculate sample statistics
mean = np.mean(data)
median = np.median(data)
mode_result = stats.mode(data)
mode = mode_result[0]  #  mode thinngy
variance = np.var(data, ddof=1)  # variance thingy
std_dev = np.std(data, ddof=1)  # standard deviation thingy
mad = np.mean(np.abs(data - mean))  # Mean Absolute Deviation thingy
coeff_variation = std_dev / mean

#zscore thingy
z_scores = np.abs(stats.zscore(data))
outliers = data[z_scores > 3]
data_cleaned = data[z_scores <= 3]

mean_cleaned = np.mean(data_cleaned)
median_cleaned = np.median(data_cleaned)
mode_result_cleaned = stats.mode(data_cleaned)
mode_cleaned = mode_result_cleaned[0]
variance_cleaned = np.var(data_cleaned, ddof=1)
std_dev_cleaned = np.std(data_cleaned, ddof=1)
mad_cleaned = np.mean(np.abs(data_cleaned - mean_cleaned))
coeff_variation_cleaned = std_dev_cleaned / mean_cleaned

# Histogram and normal distribution curve
num_bins = int(np.sqrt(len(data_cleaned)))  
plt.hist(data_cleaned, bins=num_bins, edgecolor='black', alpha=0.7, density=True)

# Normal distribution curve
x = np.linspace(min(data_cleaned), max(data_cleaned), 100)
p = stats.norm.pdf(x, mean_cleaned, std_dev_cleaned)
plt.plot(x, p, 'k', linewidth=2)
title = "Histogram and Normal Distribution Curve"
plt.title(title)
plt.xlabel('Data values')
plt.ylabel('Frequency')
plt.show()

# Evaluate 68% rule
within_one_std = np.sum((data_cleaned > mean_cleaned - std_dev_cleaned) & (data_cleaned < mean_cleaned + std_dev_cleaned))
percentage_within_one_std = within_one_std / len(data_cleaned) * 100

print("\n\n")
print("PROBLEM 14.3")
print("\n")
# Print results
print("Original Data Statistics:")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")
print(f"MAD: {mad}")
print(f"Coefficient of Variation: {coeff_variation}")
print(f"Outliers: {outliers}")

print("\nCleaned Data Statistics (without outliers):")
print(f"Mean: {mean_cleaned}")
print(f"Median: {median_cleaned}")
print(f"Mode: {mode_cleaned}")
print(f"Variance: {variance_cleaned}")
print(f"Standard Deviation: {std_dev_cleaned}")
print(f"MAD: {mad_cleaned}")
print(f"Coefficient of Variation: {coeff_variation_cleaned}")

print(f"\nPercentage of data within one standard deviation (68% rule): {percentage_within_one_std:.2f}%")

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Given data
weights = np.array([70, 75, 77, 80, 82, 84, 87, 90])
areas = np.array([2.10, 2.12, 2.15, 2.20, 2.22, 2.23, 2.26, 2.30])

# Step 1: Log transformation
log_weights = np.log(weights)
log_areas = np.log(areas)

# Step 2: Fit the linear model y = bx + c using least squares method
slope, intercept, r_value, p_value, std_err = stats.linregress(log_weights, log_areas)

# Step 3: Calculate a and b for the power law A = aW^b
b = slope
a = np.exp(intercept)

# Step 4: Predict surface area for a 95-kg person
weight_95 = 95
predicted_area_95 = a * weight_95**b

# Calculate standard error of the estimate (SEE)
predicted_log_areas = intercept + slope * log_weights
residuals = log_areas - predicted_log_areas
SEE = np.sqrt(np.sum(residuals**2) / (len(weights) - 2))

# Calculate the prediction interval
prediction_interval = 2 * SEE
predicted_area_95_upper = np.exp(np.log(predicted_area_95) + prediction_interval)
predicted_area_95_lower = np.exp(np.log(predicted_area_95) - prediction_interval)

# Step 5: Plot the data and the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(weights, areas, color='blue', label='Data points')
plt.plot(weights, a * weights**b, color='red', label='Fitted model: $A = {:.3f}W^{:.3f}$'.format(a, b))
plt.xlabel('Weight (kg)')
plt.ylabel('Surface Area (m^2)')
plt.title('Surface Area vs. Weight with Power Law Fit')
plt.legend()
plt.grid(True)
plt.savefig('surface_area_vs_weight.png')
plt.show()


#residua plottinng.
plt.figure(figsize=(10, 6))
predicted_areas = a * weights**b
residuals = areas - predicted_areas
plt.scatter(predicted_areas, residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Surface Area (m^2)')
plt.ylabel('Residuals (m^2)')
plt.title('Residuals vs. Predicted Surface Area')
plt.grid(True)
plt.savefig('residuals_vs_predicted_area.png')
plt.show()
print("\n\n")
print("PROBLEM 14.12")
print("\n")
# Print results
print(f"Power law model: A = {a:.3f}W^{b:.3f}")
print(f"Predicted surface area for 95 kg: {predicted_area_95:.2f} m^2")
print(f"Prediction interval: {predicted_area_95_lower:.2f} m^2 to {predicted_area_95_upper:.2f} m^2")

import numpy as np
import matplotlib.pyplot as plt

# Given data
nu = np.array([10, 20, 30, 40, 50, 60, 70, 80])
F = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])

# Calculate necessary sums
sum_nu = np.sum(nu)
sum_nu2 = np.sum(nu**2)
sum_nu3 = np.sum(nu**3)
sum_nu4 = np.sum(nu**4)
sum_F = np.sum(F)
sum_nuF = np.sum(nu * F)
sum_nu2F = np.sum(nu**2 * F)

# Set up the normal equations
A = np.array([[sum_nu2, sum_nu3],
              [sum_nu3, sum_nu4]])
b = np.array([sum_nuF, sum_nu2F])

# Solve for beta_1 and beta_2
beta = np.linalg.solve(A, b)
beta_1, beta_2 = beta

# Print the coefficients
print(f"beta_1 = {beta_1}")
print(f"beta_2 = {beta_2}")

# Predict the force using the fitted model
F_pred = beta_1 * nu + beta_2 * nu**2

# Plot the data and the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(nu, F, color='blue', label='Data points')
plt.plot(nu, F_pred, color='red', label='Fitted model: $F = {:.3f}\nu + {:.3f}\nu^2$'.format(beta_1, beta_2))
plt.xlabel('Velocity (m/s)')
plt.ylabel('Force (N)')
plt.title('Force vs. Velocity with Second-Order Polynomial Fit')
plt.legend()
plt.grid(True)
plt.savefig('force_vs_velocity_fit.png')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# Given data
t = np.array([15, 45, 75, 105, 135, 165, 225, 255, 285, 315, 345])
T = np.array([3.4, 4.7, 8.5, 11.7, 16, 18.7, 19.7, 17.1, 12.7, 7.7, 5.1])

# Angular frequency
omega_0 = 2 * np.pi / 365

# Design matrix
X = np.column_stack((np.ones(t.shape), np.cos(omega_0 * t), np.sin(omega_0 * t)))

# Perform linear regression
coeffs, _, _, _ = lstsq(X, T, rcond=None)
A_0, A_1, B_1 = coeffs
print("\n\n")
print("PROBLEM 15.2")
print("\n")
# Print the coefficients
print(f"A_0 = {A_0}")
print(f"A_1 = {A_1}")
print(f"B_1 = {B_1}")

# Mean temperature
mean_temp = A_0

# Amplitude
amplitude = np.sqrt(A_1**2 + B_1**2)

# Phase angle
phi = np.arctan2(B_1, A_1)

# Time of maximum temperature
t_max = (365 / (2 * np.pi)) * (-phi % (2 * np.pi))

# Maximum temperature
max_temp = A_0 + amplitude

print(f"Mean temperature: {mean_temp:.2f} °C")
print(f"Amplitude: {amplitude:.2f} °C")
print(f"Day of maximum temperature: {t_max:.2f}")
print(f"Maximum temperature: {max_temp:.2f} °C")

# Predict the temperature using the fitted model
t_fit = np.linspace(0, 365, 365)
T_fit = A_0 + A_1 * np.cos(omega_0 * t_fit) + B_1 * np.sin(omega_0 * t_fit)

# Plot the data and the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(t, T, color='blue', label='Data points')
plt.plot(t_fit, T_fit, color='red', label='Fitted model')
plt.xlabel('Day of the year')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Variation in a Pond Over the Year')
plt.legend()
plt.grid(True)
plt.savefig('temperature_fit.png')
plt.show()

import numpy as np

# Given data points
x = np.array([0, 1, 2.5, 3, 4.5, 5, 6])
y = np.array([2, 5.4375, 7.3516, 7.5625, 8.4453, 9.1875, 12])

# Select points around x = 3.5
x_selected = np.array([2.5, 3, 4.5, 5])
y_selected = np.array([7.3516, 7.5625, 8.4453, 9.1875])

# Function to calculate divided differences
def divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y

    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    
    return coef[0, :]

# Get divided difference coefficients
coefficients = divided_diff(x_selected, y_selected)

# Newton's polynomial
def newton_poly(coef, x_data, x):
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n+1):
        p = coef[n-k] + (x - x_data[n-k]) * p
    return p

# Evaluate at x = 3.5
x_val = 3.5
y_val = newton_poly(coefficients, x_selected, x_val)

print("\n\n")
print("PROBLEM 16.2")
print("\n")

print(f"Estimated value at x = 3.5 is y = {y_val:.4f}")
