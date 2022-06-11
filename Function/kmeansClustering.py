from cProfile import label
from email.policy import default
from matplotlib import pyplot as plt
import pandas as pd
from pygments import highlight
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
from cv2 import kmeans
from pyexpat import model
from sklearn.cluster import KMeans


df = pd.read_excel('DatabaseOK.xlsx')
# print(df.columns)

colum = ['สถิติ', 'การแจกแจงความน่าจะเป็นเบื้องต้น', 'ลำดับและอนุกรม',
'แคลคูลัส', 'เรขาคณิตวิเคราะห์', 'เซต', 'ตรรกศาสตร์',
'จำนวนจริงและพหุนาม', 'ฟังก์ชัน', 'ฟังก์ชันตรีโกณมิติ', 'จำนวนเชิงซ้อน',
'เมทริกซ์', 'เวกเตอร์ในสามมิติ', 'หลักการนับเบื้องต้น', 'ความน่าจะเป็น',
'ฟิสิกส์', 'เคมี', 'ชีวะ']

x = df[colum]

# labelEncode(df, colum) # X encoding

# y_le = LabelEncoder()
#y = y_le.fit_transform(df['Y'])
# x['Y_'] = y

table = pd.DataFrame()
# pca = PCA(n_components=2)
# table['x'] = pca.fit_transform(x[colum])[:, 0].astype("float16")
# table['y'] = pca.fit_transform(x[colum])[:, 1].astype("float16")
# table['Class'] = ["CS","CE","SE","IT","BC"]
n = 5
# sns.scatterplot(x=table['x'],y=table['y'],data=table,label="Class")
# plt.show()

model = KMeans(n_clusters=n)
y_Kmeans = model.fit_predict(x)
x["Y"] = y_Kmeans
x['Class'] = ["CS", "CE", "SE", "IT", "BC"]

print(x[['Y','Class']])

# sns.scatterplot(y='Y_', data=x,s=100)

# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#18 Columns
r = 18
numloc = 1
# print(model.predict([[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]]))
data_Test = [0.5,0.333333333,0.285714286,1,0.166666667,1,0.333333333,0.285714286,0.142857143,0,0,0,0,0,0,0.166666667,0.083333333,0.03125]

dataTest_df = pd.DataFrame(data=[data_Test],columns=colum)

listSubject = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tempList = []
columns = ['สถิติ', 'การแจกแจงความน่าจะเป็นเบื้องต้น', 'ลำดับและอนุกรม',
         'แคลคูลัส', 'เรขาคณิตวิเคราะห์', 'เซต', 'ตรรกศาสตร์',
         'จำนวนจริงและพหุนาม', 'ฟังก์ชัน', 'ฟังก์ชันตรีโกณมิติ', 'จำนวนเชิงซ้อน',
         'เมทริกซ์', 'เวกเตอร์ในสามมิติ', 'หลักการนับเบื้องต้น', 'ความน่าจะเป็น',
         'ฟิสิกส์', 'เคมี', 'ชีวะ','Y']

report_df = pd.DataFrame(columns=columns)

data_Test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# print(data_Test)
# print(model.predict([data_Test]))

data_Test = [0.5, 0.333333333, 0.285714286, 1, 0.166666667, 1, 0.333333333,
             0.285714286, 0.142857143, 0, 0, 0, 0, 0, 0, 0.166666667, 0.083333333, 0.03125]
# print(data_Test)
# print(model.predict([data_Test]))
# 0 -> 13

for i in range(0,r):
    listSubject[i] = 1
    predict_Y = model.predict([listSubject])
    tempList = listSubject
    tempList.append(predict_Y[0])
    report_df.loc[numloc] = tempList
    numloc = numloc + 1
    listSubject[i] = 0
    del tempList[-1]

print("========================================")
# plt.scatter(x_centroid,y_centroid,s=600,marker='x')

#Create Example Data 2 Attribute
for i in range(0,r-1):
    for j in range(i+1,r):
        listSubject[i] = 1
        listSubject[j] = 1
        predict_Y = model.predict([listSubject])
        tempList = listSubject
        tempList.append(predict_Y[0])
        report_df.loc[numloc] = tempList
        numloc = numloc + 1
        listSubject[i] = 0
        listSubject[j] = 0
        del tempList[-1]
# print(x_centroid)
# print(y_centroid)

#Create Example Data Data 3 Attribute
# print("========================================")
for i in range(r-1):
    for j in range(i+1, r-1):
        for k in range(j+1,r-2):
            listSubject[i] = 1
            listSubject[j] = 1
            listSubject[k] = 1
            predict_Y = model.predict([listSubject])
            tempList = listSubject
            tempList.append(predict_Y[0])
            report_df.loc[numloc] = tempList
            numloc = numloc + 1
            listSubject[i] = 0
            listSubject[j] = 0
            listSubject[k] = 0
            del tempList[-1]


#Create Example Data Data 4 Attribute
# print("========================================")
for i in range(r-1):
    for j in range(i+1, r-1):
        for k in range(j+2, r-2):
            for l in range(k+3,r-3):
                listSubject[i] = 1
                listSubject[j] = 1
                listSubject[k] = 1
                listSubject[l] = 1
                predict_Y = model.predict([listSubject])
                tempList = listSubject
                tempList.append(predict_Y[0])
                report_df.loc[numloc] = tempList
                numloc = numloc + 1
                listSubject[i] = 0
                listSubject[j] = 0
                listSubject[k] = 0
                listSubject[l] = 0
                del tempList[-1]

# plt.show()
# writer = pd.ExcelWriter('Report_Df_3.xlsx', engine='xlsxwriter')
# report_df.to_excel(writer, sheet_name='result', index=False)
# writer.save()

predict_Y = model.predict([data_Test])
tempList = data_Test
tempList.append(predict_Y[0])
report_df.loc[numloc] = tempList
numloc = numloc + 1

z = pd.DataFrame()
z = x[["Y","Class"]]
report_df = report_df.merge(z,how="inner")
#print(report_df[["Y","Class"]])
# print(report_df)

# report_df = report_df[colum]
# x = x[colum]
# table_ = pd.DataFrame()
# table_ = pd.concat(x)
# pca_Test = PCA(n_components=2)

# table_['x'] = pca_Test.fit_transform(report_df[colum])[:, 0]
# table_['y'] = pca_Test.fit_transform(report_df[colum])[:, 1]
table_ = pd.concat([x,report_df])
# print(table_)

# predict_Y = model.predict(table_)

# writer = pd.ExcelWriter('Report_TestCourseNew.xlsx', engine='xlsxwriter')
# table_.to_excel(writer, sheet_name='result', index=False)
# writer.save()

# sns.scatterplot(x=table['x'],y=table['y'],data=table,label="Class")
# sns.scatterplot(x=table_['x'],y=table_['y'],data=table,label="Test")
# plt.show()

