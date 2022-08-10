
from cProfile import label
from email.policy import default
from operator import mod
from matplotlib import pyplot as plt
import pandas as pd
from pygments import highlight
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
from cv2 import kmeans, rotate
from pyexpat import model
from sklearn.cluster import KMeans
import seaborn as sns


def plotGraph3D(dataModel):
    ax = plt.axes(projection='3d')
    ax.scatter(dataModel['x'], dataModel['y'], dataModel['z'], c=dataModel['Cluster'], cmap='viridis', linewidth=1,label="S")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plotGraph2D_XY(dataModel,name):
    sns.scatterplot(x=dataModel['x'],y=dataModel['y'],data=dataModel,label=name)
    
def plotGraph2D_YZ(dataModel,name):
    sns.scatterplot(x=dataModel['y'],y=dataModel['z'],data=dataModel,label=name)

def plotGraph2D_XZ(dataModel,name):
    sns.scatterplot(x=dataModel['x'],y=dataModel['z'],data=dataModel,label=name)

def printDataModel(dataModel):
    print("SE = "  + "{:.2f}%".format(len(dataModel[(dataModel['Y'] == 'SE')])/len(dataModel)))
    print("CS = "  + "{:.2f}%".format(len(dataModel[(dataModel['Y'] == 'CS')])/len(dataModel)))
    print("CE = "  + "{:.2f}%".format(len(dataModel[(dataModel['Y'] == 'CE')])/len(dataModel)))
    print("BC = "  + "{:.2f}%".format(len(dataModel[(dataModel['Y'] == 'BC')])/len(dataModel)))
    print("IT = "  + "{:.2f}%".format(len(dataModel[(dataModel['Y'] == 'IT')])/len(dataModel)))


def dropTable(cluster,tableData):
    return tableData.drop(tableData[tableData['Cluster'] != cluster].index)

df = pd.read_excel('DatabaseOK.xlsx', sheet_name="DB")
# print(df.columns)

colum = ['สถิติ', 'การแจกแจงความน่าจะเป็นเบื้องต้น', 'ลำดับและอนุกรม',
    'แคลคูลัส', 'เรขาคณิตวิเคราะห์', 'เซต', 'ตรรกศาสตร์',
    'จำนวนจริงและพหุนาม', 'ฟังก์ชัน', 'ฟังก์ชันตรีโกณมิติ', 'จำนวนเชิงซ้อน',
    'เมทริกซ์', 'เวกเตอร์ในสามมิติ', 'หลักการนับเบื้องต้น', 'ความน่าจะเป็น',
    'ฟิสิกส์', 'เคมี', 'ชีวะ']

x = df[colum]

# print(df['Y'].value_counts())
# labelEncode(df, colum) # X encoding

# y_le = LabelEncoder()
# y = y_le.fit_transform(df)
# x['Y_'] = y

table = pd.DataFrame()
pca = PCA(n_components=3)

table['x'] = pca.fit_transform(x[colum])[:, 0].astype("float16")
table['y'] = pca.fit_transform(x[colum])[:, 1].astype("float16")
table['z'] = pca.fit_transform(x[colum])[:, 2].astype("float16")
# sns.scatterplot(x=table['x'],y=table['y'],data=table,label="Class")
# sns.scatterplot(x=table['x'],y=table['y'],data=table,label="Class")
# plt.show()
n = 5

model = KMeans(n_clusters=n)
y_Kmeans = model.fit_transform(table)
print(y_Kmeans)

table['Cluster'] = model.labels_
print(table)

# table[colum] = df[colum]
table['Y'] = df['Y']
print(df)


n = [0.2,	0.333333333,	0,	0,	0,	1,	0,	0.285714286,	0.285714286,	0.5,	0,	0,	0,	0.25,	0,	0.388888889,	0.25,	0.4375]
dataTest =pd.DataFrame(columns=colum,data=[n])
print(dataTest)


dd = pca.transform([n])

print(dd)
data = pd.DataFrame(columns=['x','y','z'],data=[[1.756836 ,0.627930 ,0.871094]])
print(model.predict(data)[0])

# plotGraph3D(table)
# printDataModel(table)


result = dropTable(model.predict(data)[0],table)
printDataModel(result)
plotGraph3D(result)


# table_0 = pd.DataFrame()
# table_0 = table.drop(table[table['Cluster'] != 0].index)

# table_1 = pd.DataFrame()
# table_1 = table.drop(table[table['Cluster'] != 1].index)

# table_2 = pd.DataFrame()
# table_2 = table.drop(table[table['Cluster'] != 2].index)

# table_3 = pd.DataFrame()
# table_3 = table.drop(table[table['Cluster'] != 3].index)

# table_4 = pd.DataFrame()
# table_4 = table.drop(table[table['Cluster'] != 4].index)

# writer = pd.ExcelWriter('Report_Cluster_All.xlsx', engine='xlsxwriter')

# table.to_excel(writer, sheet_name='cluster_All', index=False)
# table_0.to_excel(writer, sheet_name='cluster_0', index=False)
# table_1.to_excel(writer, sheet_name='cluster_1', index=False)
# table_2.to_excel(writer, sheet_name='cluster_2', index=False)
# table_3.to_excel(writer, sheet_name='cluster_3', index=False)
# table_4.to_excel(writer, sheet_name='cluster_4', index=False)
# writer.save()


# Data for a three-dimensional line
# plotGraph3D(table)
# plotGraph3D(table_0)
# print("Cluster 0")
# printDataModel(table_0)
# plotGraph3D(table_1)
# print("Cluster 1")
# printDataModel(table_1)
# plotGraph3D(table_2)
# print("Cluster 2")
# printDataModel(table_2)
# plotGraph3D(table_3)
# print("Cluster 3")
# printDataModel(table_3)
# plotGraph3D(table_4)
# print("Cluster 4")
# printDataModel(table_4)
# plt.show()

# plotGraph2D_XY(table_0,"Cluster 0")
# plotGraph2D_XY(table_1,"Cluster 1")
# plotGraph2D_XY(table_2,"Cluster 2")
# plotGraph2D_XY(table_3,"Cluster 3")
# plotGraph2D_XY(table_4,"Cluster 4")
# plt.show()

# ax = plt.axes(projection='3d')
# ax.scatter(table['x'], table['y'], table['z'], c=table['Cluster'], cmap='viridis', linewidth=1);
# table['Y'] = df['Y']
# plt.show()

# print("SE = "  + "{:.2f}%".format(len(table[(table['Y'] == 'SE')])/len(table)))
# print("CS = "  + "{:.2f}%".format(len(table[(table['Y'] == 'CS')])/len(table)))
# print("CE = "  + "{:.2f}%".format(len(table[(table['Y'] == 'CE')])/len(table)))
# print("BC = "  + "{:.2f}%".format(len(table[(table['Y'] == 'BC')])/len(table)))
# print("IT = "  + "{:.2f}%".format(len(table[(table['Y'] == 'IT')])/len(table)))


# sns.scatterplot(x=table_0['x'],y=table_0['y'],data=table_0,label="Cluster 0 X,Y")
# sns.scatterplot(x=table_1['x'],y=table_1['y'],data=table_1,label="Cluster 1 X,Y")
# sns.scatterplot(x=table_2['x'],y=table_2['y'],data=table_2,label="Cluster 2 X,Y")
# sns.scatterplot(x=table_3['x'],y=table_3['y'],data=table_3,label="Cluster 3 X,Y")
# sns.scatterplot(x=table_4['x'],y=table_4['y'],data=table_4,label="Cluster 4 X,Y")
# plt.show()

# sns.scatterplot(x=table_0['y'],y=table_0['z'],data=table_0,label="Cluster 0 Y,Z")
# sns.scatterplot(x=table_1['y'],y=table_1['z'],data=table_1,label="Cluster 1 Y,Z")
# sns.scatterplot(x=table_2['y'],y=table_2['z'],data=table_2,label="Cluster 2 Y,Z")
# sns.scatterplot(x=table_3['y'],y=table_3['z'],data=table_3,label="Cluster 3 Y,Z")
# sns.scatterplot(x=table_4['y'],y=table_4['z'],data=table_4,label="Cluster 4 Y,Z")
# plt.show()

# sns.scatterplot(x=table_0['x'],y=table_0['z'],data=table_0,label="Cluster 0 X,Z")
# sns.scatterplot(x=table_1['x'],y=table_1['z'],data=table_1,label="Cluster 1 X,Z")
# sns.scatterplot(x=table_2['x'],y=table_2['z'],data=table_2,label="Cluster 2 X,Z")
# sns.scatterplot(x=table_3['x'],y=table_3['z'],data=table_3,label="Cluster 3 X,Z")
# sns.scatterplot(x=table_4['x'],y=table_4['z'],data=table_4,label="Cluster 4 X,Z")
# plt.show()
# model = KMeans(n_clusters=n)
# y_Kmeans = model.fit_predict(x)
# print(y_Kmeans)
# df['Cluster'] = model.labels_
#print(df[['Y','Cluster']])

# print(pd.crosstab(df['Y'],df['Cluster']))

# newC = df.drop(df[df['Cluster'] != 0].index)
# newD = df['Y'].tolist()

# print(newC)
# print(len(newC))
# print(newC[(newC['Y'] == 'CS')])
# print(len(newC[(newC['Y'] == 'CS')]))
# print(newC[(newC['Y'] == 'IT')])
# print(len(newC[(newC['Y'] == 'IT')]))
# print(newC[(newC['Y'] == 'BC')])
# print(len(newC[(newC['Y'] == 'BC')]))


# print("SE = "  + "{:.2f}%".format(len(newC[(newC['Y'] == 'SE')])/len(newC)))
# print("CS = "  + "{:.2f}%".format(len(newC[(newC['Y'] == 'CS')])/len(newC)))
# print("CE = "  + "{:.2f}%".format(len(newC[(newC['Y'] == 'CE')])/len(newC)))
# print("BC = "  + "{:.2f}%".format(len(newC[(newC['Y'] == 'BC')])/len(newC)))
# print("IT = "  + "{:.2f}%".format(len(newC[(newC['Y'] == 'IT')])/len(newC)))
# x["Y"] = y_Kmeans
# x['Class'] = ["CS", "CE", "SE", "IT", "BC"]

# print(x[['Y','Class']])

# sns.scatterplot(y='Y_', data=x,s=100)
# table_0 = pd.DataFrame()
# table_0 = df.drop(df[df['Cluster'] != 0].index)

# table_1 = pd.DataFrame()
# table_1 = df.drop(df[df['Cluster'] != 1].index)

# table_2 = pd.DataFrame()
# table_2 = df.drop(df[df['Cluster'] != 2].index)

# table_3 = pd.DataFrame()
# table_3 = df.drop(df[df['Cluster'] != 3].index)

# table_4 = pd.DataFrame()
# table_4 = df.drop(df[df['Cluster'] != 4].index)


# sns.scatterplot(x=table_0['สถิติ'],y=table_0['Y'],data=df,label="Class")
# sns.scatterplot(x=table_0['สถิติ'],y=table_0['การแจกแจงความน่าจะเป็นเบื้องต้น'],data=table_0,label="Cluster 0")
# sns.scatterplot(x=table_1['สถิติ'],y=table_1['การแจกแจงความน่าจะเป็นเบื้องต้น'],data=table_1,label="Cluster 1")
# sns.scatterplot(x=table_2['สถิติ'],y=table_2['การแจกแจงความน่าจะเป็นเบื้องต้น'],data=table_2,label="Cluster 2")
# sns.scatterplot(x=table_3['สถิติ'],y=table_3['การแจกแจงความน่าจะเป็นเบื้องต้น'],data=table_3,label="Cluster 3")
# sns.scatterplot(x=table_4['สถิติ'],y=table_4['การแจกแจงความน่าจะเป็นเบื้องต้น'],data=table_4,label="Cluster 4")
# plt.show()
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#18 Columns
# r = 18
# numloc = 1
# # print(model.predict([[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]]))
# data_Test = [0.5,0.333333333,0.285714286,1,0.166666667,1,0.333333333,0.285714286,0.142857143,0,0,0,0,0,0,0.166666667,0.083333333,0.03125]

# dataTest_df = pd.DataFrame(data=[data_Test],columns=colum)

# report_df = pd.DataFrame(columns=colum)

# data_Test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# print(data_Test)
# print(model.predict([data_Test]))

#data_Test = [0.5, 0.333333333, 0.285714286, 1, 0.166666667, 1, 0.333333333,
#             0.285714286, 0.142857143, 0, 0, 0, 0, 0, 0, 0.166666667, 0.083333333, 0.03125]
# print(data_Test)
# print(model.predict([data_Test]))
# 0 -> 13

# for i in range(0,r):
#     listSubject[i] = 1
#     predict_Y = model.predict([listSubject])
#     tempList = listSubject
#     tempList.append(predict_Y[0])
#     report_df.loc[numloc] = tempList
#     numloc = numloc + 1
#     listSubject[i] = 0
#     del tempList[-1]

print("========================================")
# plt.scatter(x_centroid,y_centroid,s=600,marker='x')

#Create Example Data 2 Attribute
# for i in range(0,r-1):
#     for j in range(i+1,r):
#         listSubject[i] = 1
#         listSubject[j] = 1
#         predict_Y = model.predict([listSubject])
#         tempList = listSubject
#         tempList.append(predict_Y[0])
#         report_df.loc[numloc] = tempList
#         numloc = numloc + 1
#         listSubject[i] = 0
#         listSubject[j] = 0
#         del tempList[-1]
# print(x_centroid)
# print(y_centroid)

#Create Example Data Data 3 Attribute
# print("========================================")
# for i in range(r-1):
#     for j in range(i+1, r-1):
#         for k in range(j+1,r-2):
#             listSubject[i] = 1
#             listSubject[j] = 1
#             listSubject[k] = 1
#             predict_Y = model.predict([listSubject])
#             tempList = listSubject
#             tempList.append(predict_Y[0])
#             report_df.loc[numloc] = tempList
#             numloc = numloc + 1
#             listSubject[i] = 0
#             listSubject[j] = 0
#             listSubject[k] = 0
#             del tempList[-1]


#Create Example Data Data 4 Attribute
# print("========================================")
# for i in range(r-1):
#     for j in range(i+1, r-1):
#         for k in range(j+2, r-2):
#             for l in range(k+3,r-3):
#                 listSubject[i] = 1
#                 listSubject[j] = 1
#                 listSubject[k] = 1
#                 listSubject[l] = 1
#                 predict_Y = model.predict([listSubject])
#                 tempList = listSubject
#                 tempList.append(predict_Y[0])
#                 report_df.loc[numloc] = tempList
#                 numloc = numloc + 1
#                 listSubject[i] = 0
#                 listSubject[j] = 0
#                 listSubject[k] = 0
#                 listSubject[l] = 0
#                 del tempList[-1]

# plt.show()
# writer = pd.ExcelWriter('Report_Df_3.xlsx', engine='xlsxwriter')
# report_df.to_excel(writer, sheet_name='result', index=False)
# writer.save()

# predict_Y = model.predict([data_Test])
# tempList = data_Test
# tempList.append(predict_Y[0])
#report_df.loc[len(table_)] = table_
# numloc = numloc + 1

# z = pd.DataFrame()
# z = x[["Y","Class"]]
# report_df = report_df.merge(z,how="inner")
#print(report_df[["Y","Class"]])
# print(report_df)

# report_df = report_df[colum]
# x = x[colum]
# table_ = pd.DataFrame()
# table_ = pd.concat(x)
# pca_Test = PCA(n_components=2)

# table_['x'] = pca_Test.fit_transform(report_df[colum])[:, 0]
# table_['y'] = pca_Test.fit_transform(report_df[colum])[:, 1]
# table_ = pd.concat([x,report_df])
# print(table_)

# predict_Y = model.predict(table_)

# writer = pd.ExcelWriter('Report_Cluster_0.xlsx', engine='xlsxwriter')
# table_.to_excel(writer, sheet_name='result', index=False)
# writer.save()

# sns.scatterplot(x=table['x'],y=table['y'],data=table,label="Class")
# sns.scatterplot(x=table_['x'],y=table_['y'],data=table,label="Test")
# plt.show()


