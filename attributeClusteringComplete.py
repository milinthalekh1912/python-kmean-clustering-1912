from cProfile import label
from email.policy import default
from traceback import print_tb
from unittest import result
from matplotlib import pyplot as plt
import pandas as pd
from pygments import highlight
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
from pyexpat import model
from sklearn.cluster import KMeans

df = pd.read_excel('DatabaseOK.xlsx',sheet_name="DB")

colum = ['สถิติ', 'การแจกแจงความน่าจะเป็นเบื้องต้น', 'ลำดับและอนุกรม',
'แคลคูลัส', 'เรขาคณิตวิเคราะห์', 'เซต', 'ตรรกศาสตร์',
'จำนวนจริงและพหุนาม', 'ฟังก์ชัน', 'ฟังก์ชันตรีโกณมิติ', 'จำนวนเชิงซ้อน',
'เมทริกซ์', 'เวกเตอร์ในสามมิติ', 'หลักการนับเบื้องต้น', 'ความน่าจะเป็น',
'ฟิสิกส์', 'เคมี', 'ชีวะ']

x = df[colum]
req = "BC"

model = KMeans(n_clusters=5)
y_Kmeans = model.fit_predict(x)

z = x
z["Y"] = y_Kmeans
z['Class'] = ["CS", "CE", "SE", "IT", "BC","CS", "CE", "SE", "IT", "BC","CS", "CE", "SE", "IT", "BC"]

print(z[['Y','Class']])

data_Test =[1	,1	,1	,1	,1	,1	,1	,1	,1	,1	,1	,1	,1	,1	,1	,1	,1	,1]

# print(data_Test)
result = model.predict([data_Test])
print(result)

#นำ Columns เข้าวิชาที่ต้องเรียน
rowRequest = df.loc[x["Class"] == req]

rowRequest = rowRequest.drop(["Y"], axis=1)
columnRequest = rowRequest.columns.values
classRequest = rowRequest.values.tolist()[0]

resultRequest = []

#วิชาที่ต้องเรียน
print(classRequest)

#วิชาที่รับเข้า
print(data_Test)

#ค้นหาวิชาที่ต้องเรียนเพิ่ม
for i in range(0,len(data_Test)):
    resultRequest.append(float(data_Test[i]) - float(classRequest[i]))

for i in range(0,len(columnRequest)):
    if(resultRequest[i] < 0):
        print(columnRequest[i] + " Not PASS")
    else:
        print(columnRequest[i] + " PASS")

