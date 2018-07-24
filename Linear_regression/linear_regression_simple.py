import numpy as np
def fit(x, y):
    n = len(x)
    dinominator = 0
    numerator = 0
    for i in range(0, n):
        numerator += (x[i] - np.mean(x))*(y[i] - np.mean(y))
        dinominator += (x[i] - np.mean(x))**2
    b = numerator/float(dinominator)
    a = np.mean(y)-b*float(np.mean(x))
    return a, b

def predict(x, a, b):
    return a + x*b

x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]
a, b = fit(x, y)
print("intercept:", a, " slope:", b)
print(predict(2,a,b))