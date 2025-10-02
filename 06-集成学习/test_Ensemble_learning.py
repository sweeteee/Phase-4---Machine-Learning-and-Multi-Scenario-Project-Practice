# 使用随机森林
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