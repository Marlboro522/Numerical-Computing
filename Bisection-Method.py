

def Bisection_Method() -> int: 
    """Function to find the roots of an equation using the Numerical Method of Bisection"""
    # Importing the required libraries
    import math

    # Defining the function whose roots are to be found
    def f(x: float) -> float:
        return x**3 - 2*x - 5

    # Defining the interval in which the root lies
    a = 2
    b = 3

    # Tolerable error
    e = 0.01

    # Checking if the initial values are correct
    if f(a) * f(b) > 0:
        return "Incorrect Initial Values"

    # Implementing Bisection Method
    while (b - a) >= e:
        # Finding the mid-point of the interval
        c = (a + b) / 2

        # Check if the mid-point is the root
        if f(c) == 0.0:
            break

        # Decide the side to repeat the steps
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    return c
