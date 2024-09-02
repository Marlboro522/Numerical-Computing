"Utility script to generate plots for higher order equation"
import matplotlib.pyplot as plt
import numpy as np

def eq(x):
    return -12 - (21*x) + (18 * x ** 2) - (2.75 * x ** 3)
xs = np.linspace(-10,10,10000)
ys = eq(xs)

plt.figure(figsize=(10,10))
plt.plot(xs,ys,label="Sample")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of the Higher-Order Function')
plt.legend()
plt.grid(True)
plt.show()