def regulafalsi():
    # Function to find the root of the equation
    def eq(x):
        return x**3 - x -1
    #initial Values
    x0=1
    x1=2
    e=0.0001
    while (x1-x0) >= e:
        x2 = x0 - (eq(x0) * (x1 - x0)) / (eq(x1) - eq(x0))
        if eq(x2) == 0.0:
            break
        else:
            x0 = x1
            x1 = x2
    #have to solve by hand in order to determine which value to return and when to stop. I guess x2 for now since that is the one
    #that converges? I donno
    return x2
print(regulafalsi())