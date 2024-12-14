import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the nonlinear model function
def viscosity_model(T, b0, b1):
    return b0 * np.exp(b1 / T)

# Data
T_C = np.array([26.67, 93.33, 148.89, 315.56])  # Temperatures in °C
mu = np.array([1.35, 0.085, 0.012, 0.00075])    # Viscosities in Pa·s
T_K = T_C + 273.15  # Convert to Kelvin

# Fit the model
popt, pcov = curve_fit(viscosity_model, T_K, mu)
b0, b1 = popt
standard_errors = np.sqrt(np.diag(pcov))
mu_pred = viscosity_model(T_K, *popt)
residuals = mu - mu_pred
rss = np.sum(residuals**2)
sst = np.sum((mu - np.mean(mu))**2)
r_squared = 1 - rss / sst

# Print results
print(f"Estimated b0: {b0}, b1: {b1}")
print(f"Standard errors: {standard_errors}")
print(f"R^2: {r_squared}")

# Plotting data and model fit
plt.figure(figsize=(10, 5))
plt.plot(T_K, mu, 'ro', label='Data')
plt.plot(T_K, mu_pred, 'b-', label=f'Fit: $b_0$={b0:.4f}, $b_1$={b1:.4f}')
plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (Pa·s)')
plt.title('Nonlinear Fit to Viscosity Data')
plt.legend()
plt.show()

# Plotting residuals
plt.figure(figsize=(10, 5))
plt.stem(T_K, residuals, linefmt='grey', markerfmt='D', basefmt=" ") 
plt.xlabel('Temperature (K)')
plt.ylabel('Residuals (Pa·s)')
plt.title('Residuals of the Nonlinear Fit')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True)
plt.show()
