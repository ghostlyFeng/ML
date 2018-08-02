"""
Adaboost调参：
class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)
推荐调节的参数：base_estimator,(n_estimators,learning_rate):缩小每个基分类器的贡献
"""