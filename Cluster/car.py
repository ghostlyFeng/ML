"""
分类算法应用案例-汽车金融预测用户是否会贷款买车
"""
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pydotplus

# 文件路径
data_path = 'C:/Users/Lenovo/Desktop/car.csv'

# 读取数据文件
data_frame = pd.read_csv(data_path, encoding='gbk')

# print(data_frame.head())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_frame.values[:, :-1], data_frame.values[:, -1], test_size=0.3)


# 决策树调参---1.分支依据
def adjust_criterion():
    # 参数列表
    param_dic = {'criterion':['gini', 'entropy']}
    # 构建网格搜索器
    gscv = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_dic, scoring='roc_auc')
    # 训练
    gscv.fit(X_train, y_train)

    print('best_params:{0}'.format(gscv.best_params_))
    print('best_score:{0}'.format(gscv.best_score_))


# 决策树调参---2.深度
def adjust_depth():
    # 参数列表
    param_dic = {'max_depth': range(1, 10)}
    # 构建网格搜索器
    gscv = GridSearchCV(estimator=DecisionTreeClassifier(criterion='gini'), param_grid=param_dic, scoring='roc_auc')
    # 训练
    gscv.fit(X_train, y_train)

    print('best_params:{0}'.format(gscv.best_params_))
    print('best_score:{0}'.format(gscv.best_score_))


# 决策树调参---3.最大叶子结点数
def adjust_max_leaf_nodes():
    # 参数列表
    param_dic = {'max_leaf_nodes': range(2, 100)}
    # 构建网格搜索器
    gscv = GridSearchCV(estimator=DecisionTreeClassifier(criterion='gini', max_depth=4), param_grid=param_dic, scoring='roc_auc')
    # 训练
    gscv.fit(X_train, y_train)

    print('best_params:{0}'.format(gscv.best_params_))
    print('best_score:{0}'.format(gscv.best_score_))  # 14


# KNN调参 K
def adjust_k():
    # 参数列表
    param_dic = {'n_neighbors': range(1, 20)}
    # 构建网格搜索器
    gscv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_dic, scoring='roc_auc')
    # 训练
    gscv.fit(X_train, y_train)

    print('best_params:{0}'.format(gscv.best_params_))
    print('best_score:{0}'.format(gscv.best_score_))    # 3

# 用每个算法选出来的最优参数预测并得出ROC曲线
def plot_roc(clfs):
    """
    :param clf: 分类器列表
    :return: None
    """

    for index, clf in enumerate(clfs):
        # 训练数据
        clf.fit(X_train, y_train)

        # 输出混淆矩阵
        pre = clf.predict(X_test)

        # 输出预测测试集的概率
        y_prb_1 = clf.predict_proba(X_test)[:, 1]
        # 得到误判率、命中率、门限
        fpr, tpr, thresholds = roc_curve(y_test, y_prb_1)
        # 计算auc
        roc_auc = auc(fpr, tpr)
        # 绘图
        plt.plot(fpr, tpr, label='{0}_AUC = {1:.2f}'.format(index, roc_auc))

    # 对ROC曲线图正常显示做的参数设定
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.title('ROC曲线')
    # 设置x、y轴刻度范围
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc='lower right')
    # 绘制参考线
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('命中率')
    plt.xlabel('误判率')
    plt.show()

# 输出树形图
def out_image():
    # 模型初始化
    clf = DecisionTreeClassifier(criterion='gini', max_depth=4, max_leaf_nodes=14)
    # 训练模型
    clf.fit(X_train, y_train)

    # 输出png(pdf)图形文件
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    graph.write_png(data_path.replace('.csv', '.png'))

if __name__ == '__main__':
    # adjust_criterion()
    # adjust_depth()
    # adjust_max_leaf_nodes()
    # adjust_k()
    # clfs = [DecisionTreeClassifier(criterion='gini', max_depth=4, max_leaf_nodes=14), KNeighborsClassifier(n_neighbors=3)]
    # plot_roc(clfs)
    out_image()