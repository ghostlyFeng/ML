import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
x = [4000,8000,5000,10000,9500]
y = [20000,50000,30000,70000,60000]
plt.scatter(x,y)
plt.xlabel("工资",fontproperties=font_set)
plt.ylabel("额度",fontproperties=font_set)
def fitSLR(x, y):
    n = len(x)
    dinominator = 0
    numerator = 0
    for i in range(0, n):
        numerator += (x[i] - np.mean(x))*(y[i] - np.mean(y))
        dinominator += (x[i] - np.mean(x))**2
    w = numerator/float(dinominator)
    b = np.mean(y)-w*float(np.mean(x))
    return w, b

w,b = fitSLR(x,y)
print(w,b)
y1 = w*np.float64(x)+b
plt.plot(x,y1)

plt.show()