import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('欢迎来到交互式机器学习训练场')


# load and cache raw data
@st.cache()
def get_data():
    data = load_breast_cancer()
    cancer = pd.DataFrame(data.data, columns=data.feature_names)
    cancer['target'] = data.target
    # cancer.loc[cancer.target==0,'target'] = '恶性'
    # cancer.loc[cancer.target==1,'target'] = '良性'
    return cancer


df = get_data()

# sidebar configure
st.sidebar.header('配置模型参数')
criterion = st.sidebar.selectbox(
    'Set criterion（计算分支效果的指标，可以选择基尼系数或熵）',
    ('gini', 'entropy')
)

max_depth = st.sidebar.slider(
    'Set max depth（决策树的最大深度）',
    1, 20
)

min_samples_split = st.sidebar.slider(
    'Set min samples split（能进一步往下分枝的节点最小样本数量）',
    2, 15
)

min_samples_leaf = st.sidebar.slider(
    'Set min samples leaf（每个节点要求的最小样本数）',
    2, 15
)

ccp_alpha = st.sidebar.selectbox(
    'Set cost-complexity pruning（剪枝的惩罚项参数）',
    (0, 0.001, 0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.2, 1)
)

# set display data or not
if st.checkbox('查看数据集'):
    st.dataframe(df)
    st.write('''该数据集为威斯康星乳腺癌数据集，总共569个病例，其中212个恶性，357个良性。数据集共有10个基本变量，
             代表肿瘤图片的病理参数。每个基本变量有三个维度mean, standard error, worst代表某项参数的均值，标准差和最差值，
             共计是30个特征变量。''')

st.subheader("查看特征间分布关系")
first_para = st.selectbox(
    '选择第一个变量',
    df.columns[:-1]
)
sec_para = st.selectbox(
    '选择第二个变量',
    df.columns[:-1]
)
fig, ax = plt.subplots(figsize=(8, 6))
correlation = sns.scatterplot(data=df, x=first_para, y=sec_para, hue="target", ax=ax)
st.pyplot(fig)

# build model
st.write(
    '''决策树是一种逻辑简单的机器学习算法，它是一种树形结构，使用层层推理来实现最终的分类，所以叫决策树。
    下面来看一个实际的例子。银行要用机器学习算法来确定是否给客户发放贷款，为此需要考察客户的年收入，是否有房产这两个指标。
    于是你想到了决策树模型，很快就完成了这个任务。
    首先判断客户的年收入指标。如果大于20万，可以贷款；否则继续判断。然后判断客户是否有房产。如果有房产，可以贷款；否则不能贷款。''')
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.3, stratify=df['target'],
                                                    random_state=42)
clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha,
                             random_state=42)
clf.fit(X_train, y_train)
score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

st.subheader("自己动手搭建模型")
st.write("调整左侧边栏的参数，观察模型准确率，混淆矩阵的变化。")
st.write('训练集准确率: ', round(score_train, 3))
st.write(confusion_matrix(y_train, train_predict))

st.write('测试集准确率: ', round(score_test, 3))
st.write(confusion_matrix(y_test, test_predict))
# visualzie tree
st.subheader("决策树可视化")
dot_data = export_graphviz(clf, out_file=None, feature_names=df.columns[:-1], class_names=['malignant', 'benign'],
                           rounded=True, proportion=False,
                           precision=2, filled=True)
st.graphviz_chart(dot_data)

# learning curve
X = df.iloc[:, :-1]
y = df.target
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
train_sizes = np.linspace(0.1, 1.0, 5)

st.subheader('学习率曲线')

train_sizes, train_scores, test_scores = learning_curve(clf, X=X, y=y, cv=cv, train_sizes=train_sizes)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlabel("训练样本数", fontsize=15)
ax.set_ylabel("准确率", fontsize=15)
ax.set_title("模型学习率曲线", fontsize=20)
ax.grid()
ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1,
                color="r")
ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.1,
                color="g")
ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
        label="训练集分数")
ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
        label="测试集分数")
ax.legend(loc="best", fontsize=14)
st.pyplot(fig)



