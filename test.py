def f1(a):
    # print(a **3 - 6 * a **2 + 11 * a - 6.1)
    return a **3 - 6 * a **2 + 11 * a - 6.1
# def f2(a):
#     print(3 * a **2 - 12 * a + 11)
#     return 3 * a **2 - 12 * a + 11
def result(ini):
    return ini-((f1(ini) * (0.01 * ini)) / ((f1(ini + 0.01 *ini )) - f1(ini)))
print(result(3.0467729))