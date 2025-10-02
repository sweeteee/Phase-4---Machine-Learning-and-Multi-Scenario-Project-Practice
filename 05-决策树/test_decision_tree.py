# """
# 1.【实践】泰坦尼克号生存案例
import pandas as pd
data=pd.read_csv("./titanic/train.csv")
# print(data.info())
# print(data.head())
# print(data.head(10))

x=data[['Pclass','Sex','Age']].copy()
# ‌双中括号‌：data[['Pclass', 'Sex', 'Age']]使用双中括号是为了选择多个列，返回一个DataFrame。
# 单中括号data['Age']用于选择单列，返回Series。
# copy()方法‌：copy()确实创建了一个深拷贝，修改新表不会影响原表。如果不使用copy，x可能只是原数据的一个视图，修改会影响原数据。
y=data['Survived'].copy()
x['Age'].fillna(x['Age'].mean(),inplace=True)
# x['Age']选择Age列
# fillna()方法用于填充缺失值
# x['Age'].mean()计算Age列的平均值
# inplace=True表示直接在原DataFrame上进行修改，而不返回新对象
# 完整操作：将Age列中的所有缺失值(NaN)替换为该列的平均值。

# print(x.head()) # head默认显示前五行 但head(10)可以显示10行
x=pd.get_dummies(x)#字符串值变布尔值（数值不动）
# print(x.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pre=model.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_true=y_test,y_pred=y_pre))

from sklearn.tree import plot_tree
plot_tree(model)

import matplotlib.pyplot as plt
plt.show()
# =========================================================================================

# 2.回归决策树
import numpy as np
x=np.array(list(range(1,11)))
# print(type(x)) #<class 'numpy.ndarray'>
# print(x.shape) #(10,)
x=x.reshape(-1,1)
# print(type(x)) #<class 'numpy.ndarray'>
# print(x.shape) #(10,1)
y=np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

#产生数据
from sklearn.tree import DecisionTreeRegressor
model1=DecisionTreeRegressor()

#最大深度为1的回归决策树
model1=DecisionTreeRegressor(max_depth=1)
#最大深度为3的回归决策树
model2=DecisionTreeRegressor(max_depth=3)
from sklearn.linear_model import LinearRegression
#线性回归
model3=LinearRegression()

#模型训练
model1.fit(x,y)
model2.fit(x,y)
model3.fit(x,y)

#模型预测
x_test=np.arange(0.0,10.0,0.01).reshape(-1,1)
y_pre1=model1.predict(x_test)
y_pre2=model2.predict(x_test)
y_pre3=model3.predict(x_test)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6),dpi=100)
plt.scatter(x,y,label='data')
plt.plot(x_test,y_pre1,label="max_depth=1")
plt.plot(x_test,y_pre2,label="max_depth=3")
plt.plot(x_test,y_pre3,label="linear")
plt.xlabel('data')
plt.ylabel('target')
plt.title('DecisionTreeRegressor')
plt.legend()#添加图例
plt.show()
