import pandas as pd
data=pd.read_csv("./书籍评价.csv",encoding='gbk')
# print(data.info())
content=data["内容"]
print(content.head())