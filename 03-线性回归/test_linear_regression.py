"""
# 1.实操：波士顿房价预测
from sklearn.metrics import mean_squared_error
# from sklearn.datasets import load_boston # 已经没有这个数据集了 涉及伦理问题
import numpy as np

# 1.加载数据
import pandas as pd
boston=pd.read_csv("波士顿房价xy.csv")
x=boston.iloc[:,:-1] # 除最后一列的所有数据
y=boston.iloc[:,-1] # 最后一列数据
# print(type(x)) # <class 'pandas.core.frame.DataFrame'>
# print(x)
# print(type(y)) # <class 'pandas.core.series.Series'>
# print(y)

# 2.数据集划分
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=22)

# 3.标准化
from sklearn.preprocessing import StandardScaler
process = StandardScaler()
x_train = process.fit_transform(x_train)
x_test = process.transform(x_test)

# 4.模型训练
# 4.1实例化（正规方程）

# 方法1:正规方程（Normal Equation）	一步到位，求闭式解，适合小数据集
from sklearn.linear_model import LinearRegression
# model=LinearRegression( # 创建一个线性回归模型对象，用于之后进行拟合（fit）和预测（predict）。
#     fit_intercept=True) # 是否为模型添加截距项b(True为保留)

# 方法2：随机梯度下降（SGD）	迭代求解，适合大数据或在线学习
from sklearn.linear_model import SGDRegressor
model=SGDRegressor(     # 创建一个使用**随机梯度下降法（SGD Stochastic Gradient Descent）**来拟合线性模型的回归器。
                        # 特别适用于 大数据量 或 在线学习。
                        # 本质也在拟合线性模型y=w1x1+w2x2+w3x3+...+b
    learning_rate='constant'# learning_rate='constant'：设置学习率的更新方式
                            # 这个参数控制每次更新参数时，学习率是固定还是自适应变化。
                            # 可选值有：
                            # 值	        含义
                            # 'constant'	学习率固定为 eta0（你这就是这种）
                            # 'optimal'     自适应学习率，自动根据数据选择合适初始值
                            # 'invscaling'	学习率按 η'=η0/t^p衰减
                            # 'adaptive'	误差下降就保持，误差不降就降低学习率
    ,eta0=0.01)             # 每一步的学习率都是 固定值 eta0（eta0意思是η0）
                            # 0.01 是一个比较温和的初值，在数据特征已标准化（如 StandardScaler）时通常是合理的。

# 4.2 fit
model.fit(x_train,y_train)
# print(model.coef_) # 打印训练后的模型权重（系数）(θ1,θ2,θ3,θ4,θ5,...,θn)
# print(model.intercept_) # 打印线性回归模型中的“截距项”  θ0
# print(type(model.coef_))# <class 'numpy.ndarray'>

# 4.3 打印回归模型公式
# print('y=',end='')
# for index,value in enumerate(model.coef_):
#     print(f'{value}*x{index} + ',end='')
# print(model.intercept_[0])

# 5. 预测
y_predict = model.predict(x_test)
# print(y_predict)

# 6. 模型评估
from sklearn.metrics import r2_score
r2 = r2_score(y_test , y_predict)   # R² 接近 1：模型拟合很好
                                    # R² 接近 0：模型几乎没学到什么
                                    # R² 可能为负：模型预测还不如直接用均值预测
# print("R²评分:",r2)
# ====================================================================================================
"""

"""
# 2.实践：通过代码认识过拟合和欠拟合
import numpy as np
np.random.seed(666)
x=np.random.uniform(-3,3,size=100)
# print(x.shape) #(100,)
# print(type(x)) #<class 'numpy.ndarray'>
#(1)用一元一次方程拟合（欠拟合）
X=x.reshape(-1,1) # 这么做的原因是x没法做幂运算 即x**2会报错 但X**2可以
# print(X.shape) #(100,1)
# print(type(X)) #<class 'numpy.ndarray'>
y=-x**2+x+2+np.random.normal(0,1,size=100)
# print(y.shape) #(100,)
y_test=-x**2+x+2+np.random.normal(0,1,size=100)


from sklearn.linear_model import LinearRegression
estimator=LinearRegression()
estimator.fit(X,y) # [x1,x2,x3,...,xm]和[y1,y2,y3,...,ym]拟合
y_predict=estimator.predict(X)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,y_predict,color='r')

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

plt.title("(1)用一元一次方程拟合（欠拟合）")
plt.show()

# 计算均方误差
from sklearn.metrics import mean_squared_error
print(f"一元一次多项式和训练集拟合的Mean Squared Error：{mean_squared_error(y,y_predict)}")

print(f"一元一次多项式和测试集拟合的Mean Squared Error：{mean_squared_error(y_test,y_predict)}\n")
# 通过比较正确答案和预测答案的差距 评价拟合程度

#(2)添加二次项，绘制图像 （最佳拟合）
X2=np.hstack([X,X**2])
# print(X2.shape) #(100,2)
estimator2=LinearRegression()
estimator2.fit(X2,y) # [[x1,x2,x3,...,xm],[x1^2,x2^2,x3^2,...,xm^2]]和[y1,y2,y3,...,ym]拟合
y_predict2=estimator2.predict(X2)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')# argsort(x):对 arrayx排序后，返回从小到大元素值的原索引
# plt.plot(x,y_predict2,color='r') # 为什么不能用这个？因为会变成一堆连得乱七八糟得线 为什么会变成连得乱七八糟的线？
plt.title("(2)用一元二次方程拟合（最佳拟合）")
plt.show()

print(f"一元二次多项式和训练集拟合的Mean Squared Error：{mean_squared_error(y,y_predict2)}")
print(f"一元二次多项式和测试集拟合的Mean Squared Error：{mean_squared_error(y_test,y_predict2)}\n")

#(3)再加入高次项，绘制图像，观察均方误差结果 （过拟合）
X10=np.hstack([X2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10])
estimator3=LinearRegression()
estimator3.fit(X10,y)
y_predict3=estimator3.predict(X10)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict3[np.argsort(x)],color='r')
plt.title("(3)用一元十次方程拟合（过拟合）")
plt.show()

print(f"一元十次多项式和训练集拟合的Mean Squared Error：{mean_squared_error(y,y_predict3)}")
print(f"一元十次多项式和测试集拟合的Mean Squared Error：{mean_squared_error(y_test,y_predict3)}\n")
# 可以看出训练集的MSE比测试集低 说明过拟合

# 得出结论：通过给线性模型添加二次三次项使模型泛化能力更强

# 过拟合的解决方法：正则化
# 尽量降低异常点较多的特征的影响
# print(estimator3.coef_)
# [ 1.32292089e+00  5.39520166e-01 -2.88731664e-01 -1.24760429e+00 8.06147066e-02
#   3.72878513e-01 -7.75395040e-03 -4.64121137e-02 1.84873446e-04  2.03845917e-03]

# L1正则化————Lasso 回归：会直接把高次项前面的系数变为0
from sklearn.linear_model import Lasso
estimator3_l1=Lasso(alpha=0.005)
# ,normalize=True) #将正则化强度设为0.005，查看正则化效果
# 这个错误表明在最新版本的scikit-learn中，Lasso类已经移除了normalize参数12。
# 从搜索结果来看，normalize参数在早期版本中用于控制是否对数据进行归一化处理，但现在推荐的做法是在调用fit()方法前使用sklearn.preprocessing.StandardScaler进行标准化处理
estimator3_l1.fit(X10,y)
y_predict3_l1=estimator3_l1.predict(X10)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict3_l1[np.argsort(x)],color='r')
plt.title("(4)一元十次方程拟合经过L1正则化（直接把高次项前面的系数变为0）")
plt.show()

print(f"L1正则化后的一元十次多项式和训练集拟合的Mean Squared Error：{mean_squared_error(y,y_predict3_l1)}")
print(f"L1正则化后的一元十次多项式和测试集拟合的Mean Squared Error：{mean_squared_error(y_test,y_predict3_l1)}\n")
# print(estimator3_l1.coef_)
# [ 1.04631981e+00 -8.23429179e-01 -1.61903638e-02 -4.72932628e-02 4.53284157e-03
#   4.46644747e-03  2.32154813e-04 -3.02429224e-05 -9.08112379e-05 -1.46844566e-05]
# 可以看到高次项如 θ7 θ8 θ9 θ10 变小了

# L2正则化————岭回归：把高次项前面的系数变成特别小的值
from sklearn.linear_model import Ridge
estimator3_l2=Ridge(alpha=0.005)
estimator3_l2.fit(X10,y)
y_predict3_l2=estimator3_l2.predict(X10)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict3_l2[np.argsort(x)],color='r')
plt.title("(5)一元十次方程拟合经过L2正则化（把高次项前面的系数变成特别小的值）")
plt.show()

print(f"L2正则化后的一元十次多项式和训练集拟合的Mean Squared Error：{mean_squared_error(y,y_predict3_l2)}")
print(f"L2正则化后的一元十次多项式和测试集拟合的Mean Squared Error：{mean_squared_error(y_test,y_predict3_l2)}")
# print(estimator3_l2.coef_)
# [ 1.31992256e+00  5.32877817e-01 -2.85350304e-01 -1.24199759e+00 7.94614626e-02
#   3.71205240e-01 -7.60009837e-03 -4.62056486e-02 1.77815986e-04  2.02945786e-03]
# 可以看到高次项如 θ7 θ8 θ9 θ10 没L1正则化那么小
# ====================================================================================================
"""

# 3.作业：使用L1和L2正则化方法实现波士顿房价预测

import pandas as pd
boston=pd.read_csv("波士顿房价xy.csv")
x=boston.iloc[:,:-1]
y=boston.iloc[:,-1]
# print(type(x),type(y))
# # <class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)

from sklearn.preprocessing import StandardScaler
process=StandardScaler()
x_train=process.fit_transform(x_train)
x_test=process.transform(x_test)

# L1正则化拟合
from sklearn.linear_model import Lasso
estimator_l1=Lasso(alpha=0.005)
estimator_l1.fit(x_train,y_train)
y_predict_l1=estimator_l1.predict(x_test)

import numpy as np
# print(x_test.shape,type(x_test)) # (102, 13) <class 'numpy.ndarray'>
# print(y_test.shape,type(y_test)) # (102,) <class 'pandas.core.series.Series'>
# print(np.array(y_test).shape,type(np.array(y_test))) # (102,) <class 'numpy.ndarray'>

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

plt.scatter(range(0,y_test.size),np.array(y_test)[np.argsort(y_test)])
plt.scatter(range(0,y_test.size),np.array(y_predict_l1)[np.argsort(y_test)],color='r')
plt.title("（1）使用L1正则化")
plt.show()

from sklearn.metrics import r2_score
r2 = r2_score(y_test , y_predict_l1)   # R² 接近 1：模型拟合很好
                                    # R² 接近 0：模型几乎没学到什么
                                    # R² 可能为负：模型预测还不如直接用均值预测
print("L1正则化的R²评分:",r2)
from sklearn.metrics import mean_squared_error
print(f"L1正则化和测试集拟合的Mean Squared Error：{mean_squared_error(y_test,y_predict_l1)}\n")

# L2正则化拟合
from sklearn.linear_model import Ridge
estimator_l2=Ridge(alpha=0.005)
estimator_l2.fit(x_train,y_train)
y_predict_l2=estimator_l2.predict(x_test)

plt.scatter(range(0,y_test.size),np.array(y_test)[np.argsort(y_test)])
plt.scatter(range(0,y_test.size),np.array(y_predict_l2)[np.argsort(y_test)],color='r')
plt.title("（2）使用L2正则化")
plt.show()

r2 = r2_score(y_test , y_predict_l2)   # R² 接近 1：模型拟合很好
                                    # R² 接近 0：模型几乎没学到什么
                                    # R² 可能为负：模型预测还不如直接用均值预测
print("L2正则化的R²评分:",r2)
print(f"L2正则化和测试集拟合的Mean Squared Error：{mean_squared_error(y_test,y_predict_l2)}\n")