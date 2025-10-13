# 1.kmeans API
from sklearn.datasets._samples_generator import make_blobs
# 创建数据集
X,y=make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2],random_state=9)
# n_features表示簇特征数 centers表示簇中心坐标 cluster_std表示簇方差

# 数据集可视化
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],marker='o') # marker='o'什么意思？
plt.show()

from sklearn.cluster import KMeans
# 2聚类
y_pred=KMeans(n_clusters=2,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred) # c=y_pred什么意思？
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