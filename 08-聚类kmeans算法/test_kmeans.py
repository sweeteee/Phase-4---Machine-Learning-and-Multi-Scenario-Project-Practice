# 1.kmeans API
from sklearn.datasets._samples_generator import make_blobs
# 创建数据集
X,y=make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2],random_state=9)
# n_features表示簇特征数 centers表示簇中心坐标 cluster_std表示簇方差

# 数据集可视化
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],marker='o') # marker 参数控制散点的形状（点的符号）。 'o' 表示 圆形点（circle marker）。
plt.show()

from sklearn.cluster import KMeans
# 2聚类
y_pred=KMeans(n_clusters=2,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
# plt.scatter() 画散点图时有一个参数 c（color 的缩写），表示：
# 每个点的颜色（color）。
# 这个参数既可以是：
# 一个单独的颜色值（如 'red', 'b', '#00FF00'）
# 也可以是一个数组，其中每个点对应一个颜色标签。
# 当你传入数组时（例如 c=y_pred），
# Matplotlib 会根据数组中的值自动给不同类别分配不同颜色。
plt.show()
from sklearn.metrics import calinski_harabasz_score
print("CH方法评估2聚类Kmeans的聚类分数是：",calinski_harabasz_score(X,y_pred))

# 3聚类
y_pred=KMeans(n_clusters=3,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print("CH方法评估3聚类Kmeans的聚类分数是：",calinski_harabasz_score(X,y_pred))

# 4聚类
y_pred=KMeans(n_clusters=4,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print("CH方法评估4聚类Kmeans的聚类分数是：",calinski_harabasz_score(X,y_pred))