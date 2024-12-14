import numpy as np
import matplotlib.pyplot as plt

def durand_kerner_with_convergence(coeffs, tol=1e-12, max_iter=7):
    n = len(coeffs) - 1
    roots = np.exp(2j * np.pi * np.arange(n) / n)  # Initial guesses
    convergence = []  # To store the maximum change (error) at each iteration

    for iteration in range(max_iter):
        new_roots = np.copy(roots)
        for i in range(n):
            p_i = np.polyval(coeffs, roots[i])
            prod = np.prod([roots[i] - roots[j] for j in range(n) if i != j])
            new_roots[i] = roots[i] - p_i / prod
        
        # Calculate the maximum change (error) for convergence tracking
        max_change = np.max(np.abs(new_roots - roots))
        convergence.append(max_change)
        
        # Update roots for the next iteration
        roots = new_roots
        
        # Stop if convergence tolerance is met
        if max_change < tol:
            break
        
        print(f"Iteration {iteration + 1}: {roots}")
    
    return roots, convergence

# Polynomial coefficients for x^3 - 3x^2 + 3x - 5
coeffs = [1, -3, 3, -5]
roots, convergence = durand_kerner_with_convergence(coeffs)

# Print final roots
print("Roots:", roots)

# Plot convergence
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(convergence) + 1), convergence, marker='o', linestyle='-')
plt.title("Convergence of Durand-Kerner Method")
plt.xlabel("Iteration")
plt.ylabel("Maximum Change (Error)")
plt.yscale("log")  # Log scale to show convergence clearly
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
