import math
def Bisection_Method() -> int: 
    """Function to find the roots of an equation using the Numerical Method of Bisection"""
    # Importing the required libraries

    # Defining the function whose roots are to be found
    def f(x: float) -> float:
        # return -26 + (85 * x) - (91 * x**2) + 44 * x**3 - 8 * x**4 + x**5
        return x+2

    # Defining the interval in which the root lies
    # I think this is the maximum and minimum value of the function, if we just take that it only depends on error rate how accurate our
    # roots could be. Need to think more about this. 

    a = -3.0
    b = 3.0

    # Tolerable error
    e = 0.0001

    # Checking if the initial values are correct
    # this is how we evaluate if out initial values are correct. 
    if f(a) * f(b) > 0:
        return "Incorrect Initial Values"
    i=0
    # Implementing Bisection Method
    while (b - a) >= e and i < 20:
        # Finding the mid-point of the interval
        c = (a + b) / 2

        # Check if the mid-point is the root
        if f(c) == 0.0:
            break

        # Decide the side to repeatthe steps
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        i+=1
        print(i,c,(b-a) * 100)

    return c
print(Bisection_Method())
