"""
逻辑回归调参指南
"""

from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.linear_model import LogisticRegression

data_path = r'E:\study\AI\teacher\data.csv'

# 读取数据文件
data_frame = pd.read_csv(data_path, encoding='gbk')

# 获取字段名
cols = list(data_frame.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_frame.values[:, :-1], data_frame.values[:, -1], test_size=0.3)


# 选择正则化系数
def adjust_penalty():
    # 参数列表
    param_dic = {'penalty': ['l1', 'l2']}
    # 构建网格搜索器
    gscv = GridSearchCV(estimator=LogisticRegression(), param_grid=param_dic, scoring='roc_auc')
    # 训练
    gscv.fit(X_train, y_train)

    print('best_params:{0}'.format(gscv.best_params_))
    print('best_score:{0}'.format(gscv.best_score_))    # l1


# 选择solver(依赖于penalty) : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
def adjust_solver():
    # 参数列表
    param_dic = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    # 构建网格搜索器
    gscv = GridSearchCV(estimator=LogisticRegression(penalty='l2'), param_grid=param_dic, scoring='roc_auc')
    # 训练
    gscv.fit(X_train, y_train)

    print('best_params:{0}'.format(gscv.best_params_))
    print('best_score:{0}'.format(gscv.best_score_))

# 调整C
# 注：class_weight参数用于标示分类模型中各种类型的权重，可以不输入，即不考虑权重，或者说所有类型的权重一样。
# 如果选择输入的话，可以选择balanced让类库自己计算类型权重。（不平衡问题）

if __name__ == '__main__':
    # pass
    # adjust_penalty()
    adjust_solver()