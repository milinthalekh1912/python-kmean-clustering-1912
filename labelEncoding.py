from pyexpat import model
from cv2 import kmeans
import numpy
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_excel('Attribute.xlsx',sheet_name="feature")
# ss = df["X1"].astype
# print(ss)

def labelEncode(data,columns):
    for i in columns:
        newLabel = i + "_"
        lb = LabelEncoder().fit_transform(data[i])
        data[i + "_"] = lb


colum = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11',
          'X12', 'X13', 'X14']

labelEncode(df, colum)
y_le = LabelEncoder()
y = y_le.fit_transform(df['Y'])
df['Y_'] = y
x = df[['X1_', 'X2_', 'X3_', 'X4_', 'X5_', 'X6_', 'X7_', 'X8_', 'X9_', 'X10_', 'X11_',
       'X12_', 'X13_', 'X14_']]
print(x[:5])

from sklearn.cluster import KMeans
model = KMeans(n_clusters=5)
y_Kmeans = model.fit_predict(x)
df['Cluster'] = y_Kmeans
print(df[['X1_', 'X2_', 'X3_', 'X4_', 'X5_', 'X6_', 'X7_', 'X8_', 'X9_', 'X10_', 'X11_',
          'X12_', 'X13_', 'X14_', "Y_", "Cluster"]])
