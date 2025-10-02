"""
#1.【逻辑回归】【实践】癌症分类案例
import pandas as pd
data=pd.read_csv('breast-cancer-wisconsin.csv')
# print(data.info()) # 列名 非零数值计数 数据类型

#2.数据处理
#2.1缺失值的行丢弃
import numpy as np
data=data.replace(to_replace='?',value=np.nan) # 把含有问好的列转化为not available
data=data.dropna() # 丢弃所有含有not available的行

#2.2获取特征和目标值
X=data.iloc[:,1:-1] # 病号id不要
y=data['Class']

#2.3数据划分
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)

#3.特征工程（标准化）
from sklearn.preprocessing import StandardScaler
pre=StandardScaler()
x_train=pre.fit_transform(x_train)
x_test=pre.transform(x_test)

#4.模型训练
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#5.模型预测和评估
from sklearn.metrics import accuracy_score
y_predict=model.predict(x_test)
print(accuracy_score(y_test,y_predict))

#=============================================================
"""
#2 电信客户流失预测
#【模仿一做的】
#读数据
import pandas as pd
data=pd.read_csv('churn.csv')

# 单词属性变数字方法一（replace）：yes->1,no->0,Male->1,Female->0 单词变数字
# data=data.replace(to_replace='Yes',value=1)
# data=data.replace(to_replace='No',value=0)
# data=data.replace(to_replace='Male',value=1)
# data=data.replace(to_replace='Female',value=0)

# 单词属性变数字方法二（pd.get_dummies()）单词变后缀 值变TRUE FALSE
# print(data.head())
# print(data.info())
data=pd.get_dummies(data)
# print(data.head())
# print(data.info())

#获取特征和目标值
X=data.iloc[:,1:]
y=data.iloc[:,0]
# print(X)
# print(y)

#数据划分
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)

#特征工程（标准化）
from sklearn.preprocessing import StandardScaler
pre=StandardScaler()
x_train=pre.fit_transform(x_train)
x_test=pre.transform(x_test)

#模型训练
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#模型预测
y_predict=model.predict(x_test)

#模型评估
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report
print(accuracy_score(y_test,y_predict))
print(roc_auc_score(y_test,y_predict))
#auc是roc曲线下面的面积，该值越大，模型辨别能力越强
print(classification_report(y_test,y_predict))

