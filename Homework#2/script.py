import numpy as np



########
"""Check the document once to see if all the python requirements are satisfied."""
########

# function to implement Gauss-Seidel method to solve linear system Ax = b
print("PROBELM 1")

def gauss_seidel(A, b, tol=0.05, max_iterations=100):
    x = np.zeros(len(b))
    n = len(b)
    for iteration in range(max_iterations):
        x_old = x.copy()
        for i in range(n):
            sum_ax = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_ax) / A[i][i]
        #errore computation. 
        relative_error = np.max(np.abs((x - x_old) / x))
        print(f"Iteration {iteration+1}: {x}, Relative error: {relative_error}")
        if relative_error < tol:
            break
    return x, iteration+1


A = np.array([[10, 2, -1], [-3, -6, 2], [1, 1, 5]], dtype=float)
b = np.array([27, -61.5, -21.5], dtype=float)

solution, iterations = gauss_seidel(A, b)
print(f"Solution: {solution}")
print(f"Iterations: {iterations}")
print("\n"*5)


###Concerning Eigen values. ####

#function to find Eigen values of 3x3 matrix
print("PRBOLEM 5: ")
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

print("\n"*5)

###Concerning Reactors problen####
#you got the equations, and you also need to go deeper into LU factorization and mmake it less messy to comprehend. 
###Formulation of the equations for three reaction when k is 0.1 per minute is also done, Double check the equations. 
from scipy.linalg import lu, inv
print("PROBLEM 3: ")
# Coefficients
Q_in, c_in = 10, 200
V1, V2, V3 = 50, 100, 150
Q12, Q13, Q21, Q23, Q31, Q32 = 22, 15, 10, 17, 7, 10
k = 0.1

A = np.array([[-(Q12 + Q13 + k * V1), Q21, Q31],
              [Q12, -(Q21 + Q23 + k * V2), Q32],
              [Q13, Q23, -(Q31 + Q32 + k * V3)]])
b = np.array([-Q_in * c_in, 0, 0])

# LU factorization
P, L, U = lu(A)
print("L:\n", L)
print("U:\n", U)

# Matrix inverse
A_inv = inv(A)
print("Inverse of A:\n", A_inv)

#steady sate concentations
print("\n")
c = np.dot(A_inv, b)
print("Steady-state concentrations:\n", c)
print("\n")
#concetrations with zero to reacctor 2:
b = np.array([-Q_in * c_in, 0, 0])
c = np.dot(A_inv, b)
print("Steady-state concentrations with zero flow to reactor 2:\n", c)
print("\n")

#concentration reactor 2 is halved, what will be the concentration in reactor 3?
b=np.array([-Q_in * (2 * c_in), 0, 0])
c = np.dot(A_inv, b)
print("Steady-state concentrations with concentration in reactor 2 halved:\n", c)