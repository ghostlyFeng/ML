import os
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression

data_path = r'E:\study\AI\teacher\day4\data.csv'

# 读取数据文件
data_frame = pd.read_csv(data_path, encoding='gbk')

# 获取字段名
cols = list(data_frame.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_frame.values[:, :-1], data_frame.values[:, -1], test_size=0.3)


# 分枝依据对决策树模型的影响
def adjust_criterion():
    # sklearn支持的分支依据：gini系数和信息增益
    criterions=['gini','entropy']

    for criterion in criterions:
        # 构建决策树分类器
        clf = DecisionTreeClassifier(criterion=criterion)
        # 训练模型
        clf.fit(X_train, y_train)
        # 输出模型对训练数据的预测准确率以及测试数据的预测准确率
        print(criterion,"Training score:%f" % (clf.score(X_train,y_train)))
        print(criterion,"Testing score:%f" % (clf.score(X_test, y_test)))


# 深度对决策树模型的影响
def adjust_depth():
    # 定义最大深度
    max_depth = 40

    depths = np.arange(1, max_depth)

    training_scores=[]
    testing_scores=[]

    # 对于每个深度，生成一个分类器，并添加至训练效果分数列表和测试效果分数列表
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))

    # 绘制随着决策树深度变大训练准确度和测试准确度的变化图
    plt.plot(depths,training_scores,label="traing score",marker='o')
    plt.plot(depths,testing_scores,label="testing score",marker='*')
    # 设置坐标轴选项
    plt.xlabel('max_depth')
    plt.ylabel('score')
    plt.title('Decision Tree Classification')
    # 图例
    plt.legend(loc='best')
    plt.show()


# 最大叶子节点数对决策树模型的影响
def adjust_max_leaf_nodes():
    # 定义最大叶子节点数
    max_leaf_nodes = 100

    leaf_nodes = np.arange(2, max_leaf_nodes)

    training_scores=[]
    testing_scores=[]

    # 对于每个最大叶子节点数，生成一个分类器，并添加至训练效果分数列表和测试效果分数列表
    for node in leaf_nodes:
        clf = DecisionTreeClassifier(max_leaf_nodes=node)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))

        # # AUC 值变化
        # y_prb_train_1, y_prb_test_1 = clf.predict_proba(X_train)[:, 1], clf.predict_proba(X_test)[:, 1]
        # fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_prb_test_1)
        # fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_prb_train_1)
        # roc_auc_train, roc_auc_test = auc(fpr_train, tpr_train), auc(fpr_test, tpr_test)
        #
        # training_scores.append(roc_auc_train)
        # testing_scores.append(roc_auc_test)

    # 绘制随着决策树最大叶子节点数变大训练准确度和测试准确度的变化图
    plt.plot(leaf_nodes,training_scores,label="traing score",marker='o')
    plt.plot(leaf_nodes,testing_scores,label="testing score",marker='*')
    plt.xlabel('max_leaf_nodes')
    plt.ylabel('score')
    plt.title('Decision Tree Classification')
    plt.legend(loc='best')
    plt.show()


# 输出树形图
def out_image():
    # 模型初始化
    clf = DecisionTreeClassifier(max_depth=3)
    # 训练模型
    clf.fit(X_train, y_train)
    # 输出.dot文件
    tree.export_graphviz(clf, out_file=data_path.replace('.csv', '.dot'), filled=True, rounded=True)


def out_test_file():
    # 将测试集输出到新的文件中（手动绘制ROC曲线用）
    out_path = 'test.csv'
    # 构建模型
    clf = DecisionTreeClassifier(max_depth=3)
    # 训练数据
    clf.fit(X_train, y_train)

    # 输出预测为1的概率
    prb_1 = clf.predict_proba(X_test)[:, 1]

    df_test = pd.DataFrame(X_test, columns=cols[:-1])
    # 生成预测值和预测概率
    df_test.loc[:, '营销是否成功'] = y_test
    df_test.loc[:, '预测为1的概率'] = prb_1
    # 如果文件不存在，则生成相应文件
    if not os.path.exists(out_path):
        df_test.to_csv(out_path, encoding='gbk', index=False)
    # print(df_test)


# 输出混淆矩阵和ROC曲线
def plot_roc():
    # 构建模型
    clf = DecisionTreeClassifier(max_depth=3)
    # 训练数据
    clf.fit(X_train, y_train)

    # 输出混淆矩阵
    pre = clf.predict(X_test)

    c_matrix = confusion_matrix(y_test, pre)
    # 更好的输出(二分类)
    tn, fp, fn, tp = c_matrix.ravel()
    print(c_matrix)
    print('tn={0},fp={1},fn={2},tp={3}'.format(tn,fp,fn,tp))
    # 输出预测测试集的概率
    y_prb_1 = clf.predict_proba(X_test)[:, 1]
    # 得到误判率、命中率、门限
    fpr, tpr, thresholds = roc_curve(y_test, y_prb_1)
    # 计算auc
    roc_auc = auc(fpr, tpr)

    # 对ROC曲线图正常显示做的参数设定
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 绘图
    plt.plot(fpr, tpr, 'g', label='AUC = %0.2f' % (roc_auc))
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


# 逻辑回归
def logistic_reg():

    # 正常显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 构建模型
    clf = LogisticRegression()
    # 训练数据
    clf.fit(X_train, y_train)

    # 输出混淆矩阵
    pre = clf.predict(X_test)

    c_matrix = confusion_matrix(y_test, pre)
    # 更好的输出(二分类)
    tn, fp, fn, tp = c_matrix.ravel()
    print(c_matrix)
    print('tn={0},fp={1},fn={2},tp={3}'.format(tn, fp, fn, tp))

    # 转换为矩阵
    plt.matshow(c_matrix)
    plt.title('混淆矩阵')
    # 绘制colorbar
    plt.colorbar()
    plt.ylabel('实际类型')
    plt.xlabel('预测类型')
    plt.show()


if __name__ == '__main__':
    pass
    # adjust_criterion()
    # adjust_depth()
    # adjust_max_leaf_nodes()
    # out_image()
    # plot_roc()
    out_test_file()
    # logistic_reg()