"""
# 1.使用随机森林预测泰坦尼克号生存率
import warnings
import pandas as pd
data=pd.read_csv('./titanic/train.csv')
x=data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].copy()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x['Age'].fillna(x['Age'].mean(),inplace=True)
x=pd.get_dummies(x)
# print(x)
y=data['Survived'].copy()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# 使用决策树
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
dtc_y_pre=dtc.predict(x_test)

# 使用随机森林
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(max_depth=6,random_state=9)
rfc.fit(x_train,y_train)
rfc_y_pre=rfc.predict(x_test)

from sklearn.metrics import classification_report
print('dtc_accuracy:',dtc.score(x_test,y_test))
print('dtc_report:\n',classification_report(y_true=y_test,y_pred=dtc_y_pre))
print('rfc_accuracy:',rfc.score(x_test,y_test))
print('rfc_report:\n',classification_report(y_true=y_test,y_pred=rfc_y_pre))

# 超参数选择
param={"n_estimators":[80,100,200],"max_depth":[2,4,6,8,10,12],"random_state":[9]}
from sklearn.model_selection import GridSearchCV
gc=GridSearchCV(rfc,param_grid=param,cv=2)
gc.fit(x_train,y_train)
print("使用超参数选择后，随机森林预测的准确率为",gc.score(x_test,y_test))
# ===========================================================================
"""

"""
# 2.使用 Adaboost 预测葡萄酒品质
import pandas as pd
data = pd.read_csv("./wine0501.csv")
# print(data.info())
data=data[data['Class label']!=1] # 选取class label!=1的行
x=data[['Alcohol','Hue']].copy()
y=data['Class label'].copy()
# print(y.head(100))

from sklearn.preprocessing import LabelEncoder
pre = LabelEncoder()
# 将y预处理成0~class数-1
y = pre.fit_transform(y)
# print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

# 模型训练--决策树
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=0)
tree=tree.fit(x_train,y_train)
y_train_pred=tree.predict(x_train)
y_test_pred=tree.predict(x_test)
from sklearn.metrics import accuracy_score
tree_train=accuracy_score(y_train,y_train_pred)
tree_test=accuracy_score(y_test,y_test_pred)
print("Decision tree train/test datasets accuracies are %.3f/%.3f"%(tree_train,tree_test))

# 模型训练--Adaboost
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(estimator=tree,n_estimators=500,learning_rate=0.1,random_state=0)
ada.fit(x_train,y_train)
y_train_pred=ada.predict(x_train)
y_test_pred=ada.predict(x_test)
ada_train=accuracy_score(y_train,y_train_pred)
ada_test=accuracy_score(y_test,y_test_pred)
print("Adaboost train/test datasets accuracies are %.3f/%.3f"%(ada_train,ada_test))
# =============================================================================
"""

"""
# 3.梯度提升树 Gradient Boosting Decision Tree GBDT(同时有dtc和rfc做对比)
import pandas as pd
titanic=pd.read_csv("./titanic/train.csv")
# titanic.info()
x=titanic[['Pclass','Age','Sex']].copy()
y=titanic['Survived'].copy()
x.fillna({'Age':x['Age'].mean()},inplace=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=22)

# 将数据转化为特征向量
from sklearn.feature_extraction import DictVectorizer
‘’‘
    Examples
    --------
    >> > from sklearn.feature_extraction import DictVectorizer
    >> > v = DictVectorizer(sparse=False)
    >> > D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >> > X = v.fit_transform(D)
    >> > X
    array([[2., 0., 1.],
           [0., 1., 3.]])
    >> > v.inverse_transform(X) == [{'bar': 2.0, 'foo': 1.0},
                                    ...                            {'baz': 1.0, 'foo': 3.0}]
    True
    >> > v.transform({'foo': 4, 'unseen_feature': 3})
    array([[0., 0., 4.]])
’‘’
vec=DictVectorizer(sparse=False)
# print(type(x_train)) # <class 'pandas.core.frame.DataFrame'>
# print(type(x_train.to_dict())) # <class 'dict'>
# print(type(x_train.to_dict(orient='records'))) # <class 'list'>
x_train.to_csv("./(ForDebug)beforeDictVectorizer.csv")
x_train=vec.fit_transform(x_train.to_dict(orient='records'))
# to_dict目的是将dataframe格式转化为dict格式以便于实施DictVectorizer，即将dict转化为特征向量
x_train=pd.DataFrame(x_train)
x_train.to_csv("./(ForDebug)afterDictVectorizer.csv")
x_test=vec.transform(x_test.to_dict(orient='records')) # 这里注意不能是fit_transform

# 使用单一决策树dtc进行模型训练
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)

# 使用随机森林rfc进行模型训练
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=9)
rfc.fit(x_train,y_train)

# 使用梯度提升决策树进行模型训练
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)

# 模型打分
print("Score of decision tree classifier is:",dtc.score(x_test,y_test))
print("Score of random forest classifier is:",rfc.score(x_test,y_test))
print("Score of gradient boosting classifier is:",rfc.score(x_test,y_test))

# 预测结果报告
dtc_y_pred=dtc.predict(x_test)
from sklearn.metrics import classification_report
print("Report of decision tree classifier is:\n",classification_report(dtc_y_pred,y_test))
rfc_y_pred=rfc.predict(x_test)
print("Report of random forest classifier is:\n",classification_report(rfc_y_pred,y_test))
gbc_y_pred=gbc.predict(x_test)
print("Report of gradient boosting classifier is:\n",classification_report(rfc_y_pred,y_test))

# ============================================================================================
"""

# 4. XGboost 红酒品质预测
import pandas as pd
data=pd.read_csv("./红酒品质分类.csv")
x=data.iloc[:,:-1]
y=data.iloc[:,-1]-3
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2)
# stratify=y 表示分层抽样：
# 保证每个品质类别在训练集和测试集中都有相同比例的数据，避免样本不均衡导致某些类别太少。
from sklearn.utils import class_weight
class_weight=class_weight.compute_sample_weight(class_weight='balanced',y=y_train)
# compute_sample_weight() 会根据类别出现的次数计算每个样本的“权重”。
# class_weight='balanced' 让模型在训练时更关注少数类别，防止出现“模型只会预测多数类”的问题。
from xgboost import XGBClassifier
model=XGBClassifier(n_estimators=5,objective='multi:softmax')
# XGBClassifier 是一个基于梯度提升（Gradient Boosting）的强大分类算法。
# n_estimators=5：只训练 5 棵树（树越多越准确，但计算慢，这里只是小规模示范）。
# objective='multi:softmax'：多分类任务，输出为整数类别（非概率）。
model.fit(x_train,y_train,sample_weight=class_weight)
# sample_weight：每个样本的权重，前面计算得到，用于平衡类别。
y_pre=model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pre))