import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given data
T = np.array([273, 283, 293, 303, 313, 323, 333])
mu = np.array([1.79, 1.51, 1.29, 1.11, 0.97, 0.85, 0.75])

# Define the nonlinear model
def nonlinear_model(T, b0, b1):
    return b0 * np.exp(b1 / T)

# Perform nonlinear regression
"""popt: Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized."""
"""pcov: The estimated covariance of popt. The diagonals provide the variance of the parameter estimate."""
popt, pcov = curve_fit(nonlinear_model, T, mu)

# Extract the parameters
b0, b1 = popt

# Calculate predicted values and residuals
mu_pred = nonlinear_model(T, b0, b1)
residuals = mu - mu_pred

# Calculate R^2
ss_res = np.sum(residuals**2)
ss_tot = np.sum((mu - np.mean(mu))**2)
r2 = 1 - (ss_res / ss_tot)

# Standard error of the estimate
se = np.sqrt(ss_res / (len(mu) - 2))

# Standard error estimates for the parameters
perr = np.sqrt(np.diag(pcov))

# Print the results
print(f"Estimated parameters: b0 = {b0:.4f}, b1 = {b1:.4f}")
print(f"Standard errors: b0 = {perr[0]:.4f}, b1 = {perr[1]:.4f}")
print(f"R^2: {r2:.4f}")
print(f"Standard error of the estimate (se): {se:.4f}")

# Plot the data and the model
plt.figure(figsize=(10, 5))
plt.scatter(T, mu, label='Data', color='blue')
plt.plot(T, mu_pred, label='Fitted model', color='red')
plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (Pa·s)')
plt.title('Viscosity vs Temperature')
plt.legend()
plt.grid(True)
plt.savefig('viscosity_model.png')
plt.show()

# Plot residuals vs predicted values
plt.figure(figsize=(10, 5))
plt.scatter(mu_pred, residuals, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Viscosity (Pa·s)')
plt.ylabel('Residuals (Pa·s)')
plt.title('Residuals vs Predicted Viscosity')
plt.grid(True)
plt.savefig('residuals.png')
plt.show()
