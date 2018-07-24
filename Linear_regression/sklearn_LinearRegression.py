from sklearn import linear_model
clf = linear_model.LinearRegression()
X = [[0,0],[1,1],[2,2]]
y = [0,1,2]
clf.fit(X,y)
print(clf.coef_)
print(clf.intercept_)
"""
sklearn一直秉承着简洁为美得思想设计着估计器，
实例化的方式很简单，使用clf = LinearRegression()就可以完成，
但是仍然推荐看一下几个可能会用到的参数：

fit_intercept：是否存在截距，默认存在
normalize：标准化开关，默认关闭

回归

其实在上面的例子中已经使用了fit进行回归计算了，使用的方法也是相当的简单。

fit(X,y,sample_weight=None)：X,y以矩阵的方式传入，
而sample_weight则是每条测试数据的权重，同样以array格式传入。
predict(X)：预测方法，将返回预测值y_pred
score(X,y,sample_weight=None)：评分函数，将返回一个小于1的得分，
可能会小于0
方程

LinearRegression将方程分为两个部分存放，coef_存放回归系数，
intercept_则存放截距，因此要查看方程，就是查看这两个变量的取值。
"""