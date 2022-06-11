from cProfile import label
from pyexpat import model
from traceback import print_tb
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


colum = ['สถิติ', 'การแจกแจงความน่าจะเป็นเบื้องต้น', 'ลำดับและอนุกรม',
         'แคลคูลัส', 'เรขาคณิตวิเคราะห์', 'เซต', 'ตรรกศาสตร์',
         'จำนวนจริงและพหุนาม', 'ฟังก์ชัน', 'ฟังก์ชันตรีโกณมิติ', 'จำนวนเชิงซ้อน',
         'เมทริกซ์', 'เวกเตอร์ในสามมิติ', 'หลักการนับเบื้องต้น', 'ความน่าจะเป็น',
         'ฟิสิกส์', 'เคมี', 'ชีวะ']

df = pd.read_excel('Attribute.xlsx', sheet_name="DB")
table = pd.DataFrame()

pca = PCA(n_components=4)
table['x'] = pca.fit_transform(df[colum])[:, 0]
table['y'] = pca.fit_transform(df[colum])[:, 1]
table['z'] = pca.fit_transform(df[colum])[:, 2]
table['a'] = pca.fit_transform(df[colum])[:, 3]

model = KMeans(n_clusters=5)
y_Kmeans = model.fit_predict(table)
table['Cluster'] = y_Kmeans
table['Class'] = df['Y']
print(table)

data_Test = [[0.5, 0.333333333, 0.285714286, 1, 0.166666667, 1, 0.333333333,
             0.285714286, 0.142857143, 0, 0, 0, 0, 0, 0, 0.166666667, 0.083333333, 0.03125],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

_tableTest = pd.DataFrame(data=data_Test,columns=colum)

# ind = _tableTest.shape[0] + 1
# _tableTest.loc[ind] = data_Test

# ind = _tableTest.shape[0] + 1
# _tableTest.loc[ind] = data_Test_1

_Test = pd.DataFrame()
_Test["x"] = pca.transform(_tableTest[colum])[:, 0]
_Test["y"] = pca.transform(_tableTest[colum])[:, 1]
_Test["z"] = pca.transform(_tableTest[colum])[:, 2]
_Test["a"] = pca.transform(_tableTest[colum])[:, 3]

y_Test = model.predict(_Test)
_Test['Cluster'] = y_Test
print(_Test)

sns.scatterplot(x=_Test['x'], y=_Test['y'], data=_Test,color="black")
sns.scatterplot(x=_Test['z'], y=_Test['a'], data=_Test, color="pink")

sns.scatterplot(x=table['x'], y=table['y'], data=table,hue=table['Class'])
sns.scatterplot(x=table['z'], y=table['a'], data=table,hue=table['Class'])

plt.show()