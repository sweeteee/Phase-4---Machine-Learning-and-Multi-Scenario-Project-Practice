"""
# 1 分类与回归

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# x=[[0,2,3],[1,3,4],[3,5,6],[4,7,8],[2,3,4]]
# y=[0,0,1,1,0]
x=[[0,1,2],[1,2,3],[2,3,4],[3,4,5]]
y=[0.1,0.2,0.3,0.4]

# model=KNeighborsClassifier(n_neighbors=3)
model=KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
print(model.predict([[4,4,5]]))

====================================================================================================
"""

"""
# 2 归一化预处理
from sklearn.preprocessing import MinMaxScaler,StandardScaler
x=[[90,2,10,40],[60,4,15,45],[75,3,13,46]]

# 归一化 通过对原始数据进行变换把数据映射到【mi,mx】(默认为[0,1])之间
process=MinMaxScaler()
data=process.fit_transform(x)
print(data)

# 标准化 通过对原始数据进行标准化，转换为均值为0标准差为1的标准正态分布的数据
process=StandardScaler()
data=process.fit_transform(x)
print(data)

print(process.mean_) # 每个特征的均值
print(process.var_) # 每个特征的方差
====================================================================================================
"""

"""
# 3 【实操】利用KNN算法进行鸢尾花分类

# 加载工具包
from sklearn.datasets import load_iris

import seaborn as sns
# Seaborn 是一个基于 Matplotlib 的 Python 数据可视化库，专注于统计数据的可视化。
# 它通过简化绘图流程、优化默认样式和提供高级接口，帮助用户更高效地创建美观且信息丰富的统计图表。

import matplotlib.pyplot as plt

import pandas as pd
# Pandas 是 Python 中最流行的数据分析和处理库，专门设计用于高效处理结构化数据（表格数据）。
# 它提供了灵活的数据结构和工具，使得数据清洗、转换、分析和可视化变得简单高效。

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
iris_data = load_iris()
# print(iris_data)
# print(type(iris_data))
# # 说它是一个bunch类 但我感觉更贴近字典
# print(iris_data.keys())
# # data是每项属性值 target是每项的标签
# # feature_names是每项的属性名 target_names是每项的标签名
# print(iris_data.feature_names)
# print(iris_data.target_names)
# print(iris_data.data_module)

# pandas是一个专门用于数据挖掘的python库，它以Numpy为基础，借力Numpy模块在计算机方面性能强的优势，
# 而且基于matplotlib，能够方便的画图，还拥有自己独特的数据结构，如DataFrame和Series
iris_df = pd.DataFrame(iris_data['data'],columns=iris_data.feature_names)
# # 把属性值和属性名读进iris_df
# print(type(iris_df))
# print(iris_df.keys())

iris_df['label']=iris_data.target
# # 将标签值添加进iris_df
# print(iris_df.keys())

sns.lmplot(x='sepal length (cm)',y='sepal width (cm)',data=iris_df,hue='label')
# 一个空格都不能错
# 默认会同时显示 散点图 和 线性回归拟合线
# 不同label类别的数据会以不同颜色显示
# 自动添加 回归方程 和 置信区间带
# plt.show()

# 特征工程

# 数据集划分（训练集和测试集）
# x是输入y是输出
x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.3,random_state=22)
# print(x_train) #105
# print(f"Len of x_train is:{len(x_train)}")
# print(y_train) #105
# print(f"Len of y_train is:{len(y_train)}")
# print(x_test) #45
# print(f"Len of x_test is:{len(x_test)}")
# print(y_test) #45
# print(f"Len of y_test is:{len(y_test)}")

# 标准化
# 使用正交标准化 实例化为一个对象process
process = StandardScaler()
x_train = process.fit_transform(x_train)
# 先用fit计算数据的统计量（如均值、方差），然后用fit的统计量通过transform进行标准化转换
# print(x_train)
x_test = process.transform(x_test)
# 直接用上面的fit的统计量通过transform进行标准化转
# print(x_test)

# 训练模型
# 实例化KNN模型为model
model = KNeighborsClassifier(n_neighbors=3)

# 调用fit法
model.fit(x_train,y_train)
# 通过训练数据（特征x_train和标签y_train）学习模型参数，建立特征与目标变量之间的映射关系
# 对于KNN分类器，该过程会存储所有训练样本的特征向量和标签，用于后续预测时的距离计算
# 和StandardScaler()中的fit不一样
# 预处理中的fit()：仅计算数据的统计量（如均值、标准差），不涉及标签信息
# 模型中的fit()：同时利用特征和标签进行有监督学习，生成决策规则

# 技术实现特点
# KNN算法的fit()方法属于惰性学习（lazy learning），实际不进行显式计算，仅保存训练数据集。
# 预测阶段才会动态计算新样本与存储样本的距离。

# 流程必要性
# 该步骤必须出现在transform()之后，确保模型接收的是经过标准化的数据，避免特征量纲差异对距离计算的影响。
# 标准化后的数据能提升KNN等基于距离的算法性能。


x=[[5.1,3.5,1.4,0.2]]
x=process.transform(x)
# 直接用之前的fit的统计量通过transform进行标准化转
# print(x)
# print(model.predict_proba(x))
# 判断x属于各个target_name的概率

# 模型评估（准确率）
# 6.1 使用预测结果y_predict
y_predict=model.predict(x_test)
# 根据训练好的模型对测试集进行预测
# print(y_predict)
acc = accuracy_score(y_test,y_predict)
print(acc)

# 6.2 直接给x_test，y_test测试模型准确性
acc = model.score(x_test,y_test)
print(acc)
====================================================================================================
"""

"""
# 4【实操】 交叉网格验证搜索在鸢尾花分类中的应用
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
# train_test_split 是 scikit-learn 中的一个函数，作用是：
# 把一个完整的数据集分成“训练集”和“测试集”，通常用于机器学习中的模型训练与评估。
# GridSearchCV 是 scikit-learn 中的一个自动调参工具，用于在多个参数组合中找到最优组合。
# Cross Validation，交叉验证，确保结果可靠
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1加载数据
data=load_iris()
# print(type(data))
# print(data.keys())
# print(data.data)
# print(f'\n{data.target}')
# 2数据集划分
x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=22)
# 3特征预处理
pre=StandardScaler()
x_train=pre.fit_transform(x_train)
# 4模型实例化+交叉验证+网格搜索
model=KNeighborsClassifier(n_neighbors=1)
#创建一个K近邻分类器（KNN）
# 初始设定K=1（即每次预测看“最近的1个邻居”）
# 🧠说明：
# 虽然这里设置了n_neighbors=1，但其实后面你用 GridSearchCV 会覆盖这个值，它只是在初始化时的默认值。
paras_grid={'n_neighbors':[4,5,7,9]}
# 创建了一个“参数字典”，告诉 GridSearchCV 要尝试哪几个参数组合(K值（邻居个数）)。
estimator=GridSearchCV(
    estimator=model,        # 你要优化的基础模型（KNN）
    param_grid=paras_grid,  # 要搜索的参数范围（上面那张表）
    cv=4)                   # 使用4折交叉验证
# 创建一个 网格搜索 + 交叉验证（CrossValidation）的模型选择器，名字叫 estimator。
# 它会在参数范围 {n_neighbors: [4, 5, 7, 9]} 中进行遍历，
# 并对每个参数使用 4折交叉验证，从中找出表现最好的那个参数。
########################################
# 以下过程仅当你喂给它数据后才会发生
# （在本程序中是：
# estimator.fit(x_train,y_train)
# ）
# 每次用 3 个做训练，1 个做验证，共进行 4 次训练/验证，然后取平均分。
# 训练：1+2+3 → 验证：4
# 训练：1+2+4 → 验证：3
# 训练：1+3+4 → 验证：2
# 训练：2+3+4 → 验证：1
# 这个过程对 4、5、7、9 每个 K 值都执行一遍。
# 执行完之后，你可以获得：
# 属性                       含义
# estimator.best_params_    最优参数（比如：{'n_neighbors': 5}）
# estimator.best_score_	    最优参数对应的平均交叉验证分数(满分1.0，对应100%准确度)
# estimator.best_estimator_	用最优参数训练好的模型
# estimator.predict(X_test)	直接用最优模型预测
########################################
# print(type(estimator))
estimator.fit(x_train,y_train)
# 开始执行网格搜索：用 x_train 和 y_train 数据，
# 针对每个 n_neighbors 值，进行 4 次交叉验证，最后找到最优的参数，并把最优模型训练出来。

# print(estimator.best_params_)
print(estimator.best_score_)
# print(estimator.best_estimator_) # 准确率0.9666666666666668
# print(estimator.cv_results_) #包含所有组合训练结果的字典

model=KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)
# x=[[5.1,3.5,1.4,0.2]]
# x=pre.transform(x)
y_predict=model.predict(x_test)
print(accuracy_score(y_test,y_predict)) #准确率0.4666666666666667 远低于 GridSearchCV
====================================================================================================
"""

"""
#5 编写KNN代码实现手写数字识别
#（特征预处理，交叉验证网格搜索）
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

#获取数据
data=pd.read_csv("手写数字识别.csv")
# print(data.keys()) # label指标签，分为0-9 总共784个像素，即28*28
# print(min(data['pixel400'])，max(data['pixel400'])) # 每个像素的灰度范围为0-255
x=data.iloc[: , 1:]
# 选取所有的行 选取从下标为1开始所有的列
# iloc（全称：integer-location based indexing） 是 pandas 中用于按“位置”选择数据的方法。

y=data.iloc[: , 0]
# 选取所有的行 选取下标为0的列

# 特征归一化 【注意】
transform=MinMaxScaler()
# print(type(x))
# print(x.iloc[:,399])
# print(x.iloc[1,399])
# print(x.iloc[1,399]/255) # MinMaxScaler 在这里等价于/255

x=transform.fit_transform(x)
# print(type(x))
# print(x[:,399])
# print(x[1,399])

# 数据集划分
x_train,x_test,y_train,y_test=train_test_split(x,        #x：特征数据，归一化之后的每张图片的784个像素值（已经用 MinMaxScaler 处理过了）。
                                               y,               #y：标签数据，也就是每张图代表的数字（0-9）。
                                               test_size=0.2,   #表示测试集占总数据的 20%，训练集占 80%。
                                               stratify=y,      #这是一个非常重要的参数，表示：按原始标签的比例，进行“分层抽样”划分数据。
                                                                #如果你不写 stratify=y，系统是「完全随机」选出20%的数据作为测试集，
                                                                #有可能某些标签的数量非常少，甚至没有，导致测试结果不准确。
                                                                #写了 stratify=y 之后，划分出来的训练集和测试集中，
                                                                #每个类别的比例会和原始数据中一样，避免类别不平衡问题。
                                               random_state=22) #用于设置「随机种子」，让你每次运行代码时划分出的训练集/测试集是一样的。
#模型实例化
model=KNeighborsClassifier(n_neighbors=1)
#网格搜索交叉验证
param_grid={'n_neighbors':[3,5,7,9,10,11]}
model=(GridSearchCV             # 这是 scikit-learn 中的一个函数，全称是：
                                # Grid Search Cross Validation（网格搜索交叉验证）
                                # 它的作用是：
                                # 在你设置的一堆“参数组合”中，通过交叉验证自动找出效果最好的参数组合，并返回对应的模型。
       (estimator=model,        # estimator 是你想要调参的模型，必须是一个已经实例化过的模型对象。
        param_grid=param_grid,  # param_grid 是一个字典，你要搜索的参数范围。
        cv=4,                   # 这是 交叉验证（Cross Validation） 的折数，意思是：
                                # 每种参数组合都用4折交叉验证来评估性能。
        verbose=2))             #在英文中：verbose = “话多的”，“啰嗦的”
                                # 在编程中它的意思是：控制程序运行过程中要不要“说话”，说多少话。
                                # 换句话说：verbose 参数决定程序在执行时是否输出中间信息，比如进度、细节、训练情况等等。
                                # 0（默认）	安静模式	什么都不输出
                                # 1	        简略模式	只显示每一组参数的整体训练开始/完成
                                # 2	        详细模式	显示每一折交叉验证的训练情况
                                # 3 及以上	超详细	会显示更细粒度的信息（几乎不会用到）

# print(f"type of x_train is {type(x_train)}") # <class 'numpy.ndarray'> 可以进行训练
# print(f"type of y_train is {type(y_train)}") # <class 'pandas.core.series.Series'> 可以进行训练
model.fit(x_train,y_train)
# print(model.best_estimator_)    # 得出3的准确率最高

# 模型训练
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

# 模型预测
img=plt.imread('demo.png')
img=img.reshape(1,-1)
# 这是在对图像进行形状变换（reshape），准备送入模型。
# 解读：.reshape(1, -1) 表示把图片变成一维向量并加上一个批次维度；
# 假设原图是 28×28，变成 (1, 784)；
# 也就是说，它现在变成了一个 1 行、784 列 的二维数组，符合 KNN 模型输入格式。
img=transform.transform(img) # MinMaxScaler
y_pred=model.predict(img)
print(f"预测结果为：{y_pred[0]}")

# 模型评估
print(model.score(x_test,y_test))
# ====================================================================================================
"""

"""
#6 Kaggle竞赛实战
import pandas as pd
train=pd.read_csv("kaggle/train.csv")#kaggle前面不能有/
x=train.iloc[:,1:]
# print(x)
y=train.iloc[:,0]
# print(y)



from sklearn.preprocessing import MinMaxScaler
transform=MinMaxScaler()
x=transform.fit_transform(x)

from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier(n_neighbors=1)
# param_grid={'n_neighbors':[3,5,7,9,10,11]}
# from sklearn.model_selection import GridSearchCV
# model=GridSearchCV(estimator=model,param_grid=param_grid,cv=4,verbose=2)
# model.fit(x,y)
# print(model.best_estimator_)# 得出3的准确率最高

model=KNeighborsClassifier(n_neighbors=3)
# print(f"type of x is {type(x)}") # <class 'numpy.ndarray'> 可以进行训练
# print(f"type of y is {type(y)}") # <class 'pandas.core.series.Series'> 可以进行训练
model.fit(x,y)
print(model.score(x,y))

test=pd.read_csv("kaggle/test.csv")
x_test=test.iloc[:,:]
# print(x_test)
x_test=transform.transform(x_test)
y_predict=model.predict(x_test)

#生成submission文件
import numpy as np
submission=pd.DataFrame({"ImageId":np.arange(1,len(y_predict)+1),
                         "Label":y_predict})

submission.to_csv("kaggle/submission.csv",
                  index=False)  # index=False表示：
                                # 不要把 DataFrame 的行索引写进 CSV 文件（否则 CSV 的第一列会变成 0、1、2、3……）
"""