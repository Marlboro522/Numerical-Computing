import numpy as np

def gauss_seidel(A, b, tol=0.05, max_iterations=100):
    x = np.zeros(len(b))
    n = len(b)
    for iteration in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            sum_ax = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_ax) / A[i][i]
        # Compute relative error
        relative_error = np.max(np.abs((x - x_old) / x))
        if relative_error < tol:
            break
    return x, iteration+1

A = np.array([[10, 2, -1], [-3, -6, 2], [1, 1, 5]], dtype=float)
b = np.array([27, -61.5, -21.5], dtype=float)

solution, iterations = gauss_seidel(A, b)
print(f"Solution: {solution}")
print(f"Iterations: {iterations}")


#function to find Eigen values of 3x3 matrix

def eigen_values(A):
    # compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues
A = np.array([[20, 3, 2], [3, 9, 4], [2, 4, 12]], dtype=float)
eigenvalues = eigen_values(A)
print(f" The Eigenvalues: {eigenvalues}")

def power_method(A, num_iterations):
    #randdomizing for showing improvement over iteratiosnn,. 
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k1_norm, b_k

def inverse_power_method(A, num_iterations, tol=1e-7):
    A_inv = np.linalg.inv(A)
    eigenvalue, eigenvector = power_method(A_inv, num_iterations)
    eigenvalue = 1 / eigenvalue
    return eigenvalue, eigenvector

A = np.array([[20, 3, 2],
              [3, 9, 4],
              [2, 4, 12]])

# Power method to find the largest eigenvalue
num_iterations = 20
largest_eigenvalue, _ = power_method(A, num_iterations)
print(f"Largest eigenvalue Estimation(Power method): {largest_eigenvalue}")

# Inverse power method to find the smallest eigenvalue
smallest_eigenvalue, _ = inverse_power_method(A, num_iterations)
print(f"Smallest eigenvalue Estimation(Inverse power method): {smallest_eigenvalue}")
