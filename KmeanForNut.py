import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel('Attribute.xlsx', sheet_name="DB")

colum = ['สถิติ', 'การแจกแจงความน่าจะเป็นเบื้องต้น', 'ลำดับและอนุกรม',
         'แคลคูลัส', 'เรขาคณิตวิเคราะห์', 'เซต', 'ตรรกศาสตร์',
         'จำนวนจริงและพหุนาม', 'ฟังก์ชัน', 'ฟังก์ชันตรีโกณมิติ', 'จำนวนเชิงซ้อน',
         'เมทริกซ์', 'เวกเตอร์ในสามมิติ', 'หลักการนับเบื้องต้น', 'ความน่าจะเป็น',
         'ฟิสิกส์', 'เคมี', 'ชีวะ']

x = df[colum]

n = 5
model = KMeans(n_clusters=n)

y_Kmeans = model.fit_predict(x)
x["Y"] = y_Kmeans
x['Class'] = ["CS", "CE", "SE", "IT", "BC"]

r = 18
numloc = 1

data_Test = [0.5, 0.333333333, 0.285714286, 1, 0.166666667, 1, 0.333333333,
             0.285714286, 0.142857143, 0, 0, 0, 0, 0, 0, 0.166666667, 0.083333333, 0.03125]

dataTest_df = pd.DataFrame(data=[data_Test], columns=colum)

listSubject = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tempList = []
columns = ['สถิติ', 'การแจกแจงความน่าจะเป็นเบื้องต้น', 'ลำดับและอนุกรม',
           'แคลคูลัส', 'เรขาคณิตวิเคราะห์', 'เซต', 'ตรรกศาสตร์',
           'จำนวนจริงและพหุนาม', 'ฟังก์ชัน', 'ฟังก์ชันตรีโกณมิติ', 'จำนวนเชิงซ้อน',
           'เมทริกซ์', 'เวกเตอร์ในสามมิติ', 'หลักการนับเบื้องต้น', 'ความน่าจะเป็น',
           'ฟิสิกส์', 'เคมี', 'ชีวะ', 'Y']

report_df = pd.DataFrame(columns=columns)

print("================= 1 Atribute =======================")

for i in range(0, r):
    listSubject[i] = 1
    predict_Y = model.predict([listSubject])
    tempList = listSubject
    tempList.append(predict_Y[0])
    report_df.loc[numloc] = tempList
    numloc = numloc + 1
    listSubject[i] = 0
    del tempList[-1]

print("================= 2 Atribute =======================")

for i in range(r-1):
    for j in range(i, r-1):
        listSubject[i] = 1
        listSubject[j+1] = 1
        predict_Y = model.predict([listSubject])
        tempList = listSubject
        tempList.append(predict_Y[0])
        report_df.loc[numloc] = tempList
        numloc = numloc + 1
        listSubject[i] = 0
        listSubject[j+1] = 0
        del tempList[-1]

print("================= 3 Attribute ==========================")
for i in range(r-1):
    for j in range(i, r-1):
        for k in range(j, r-2):
            listSubject[i] = 1
            listSubject[j+1] = 1
            listSubject[k+2] = 1
            predict_Y = model.predict([listSubject])
            tempList = listSubject
            tempList.append(predict_Y[0])
            report_df.loc[numloc] = tempList
            numloc = numloc + 1
            listSubject[i] = 0
            listSubject[j+1] = 0
            listSubject[k+2] = 0
            del tempList[-1]

predict_Y = model.predict([data_Test])
tempList = data_Test
tempList.append(predict_Y[0])
report_df.loc[numloc] = tempList
numloc = numloc + 1

z = pd.DataFrame()
z = x[["Y", "Class"]]
report_df = report_df.merge(z, how="inner")

table_ = pd.concat([x, report_df])
print(table_)
