import pandas as pd
data=pd.read_csv("./书籍评价.csv",encoding='gbk')
data['labels']=np.where(data['评价']=='好评',1,0)