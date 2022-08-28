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
import matplotlib.pyplot as plt
import seaborn as sns


def labelEncode(data, columns):
    for i in columns:
        newLabel = i + "_"
        lb = LabelEncoder().fit_transform(data[i])
        data[i + "_"] = lb

def percentForMajor(table,major,user):
    listMajor = table[(table['Y'] == major)].drop(columns='Y').values[0]
    print("In Function percentForMajor")
    result = []
    for i in range(0,6):
        print(str(user[i]) + " - " + str(listMajor[i]))
        result.append(user[i] - listMajor[i])
    print(result)
    return result.count(0)/6




df = pd.read_excel('Aptitute.xlsx', sheet_name="CaseDss")

colum = ["Coding", "Artificial Intelligence","Web","Programming","Business","Engineer"]


x = df[colum]

table = pd.DataFrame()

n = 5

model = KMeans(n_clusters=n)
y_Kmeans = model.fit_predict(x)
x["Y"] = y_Kmeans
# x['Class'] = ["CS", "CE", "SE", "IT", "BC"]
x['Class'] = df['Y']
# x['Class'] = ["CS", "CE", "SE", "IT", "BC", "CS",
#               "CE", "SE", "IT", "CS", "CE", "SE", "IT", "CS", "CE", "SE", "IT", "CS", "CE", "SE", "IT"
#               ]
csMajor = df[(df['Y'] == 'CS')]
csMajor = csMajor.drop(columns='Y')
print(csMajor.values[0])
# print("Result CS : " + str(percentForMajor(csMajor.values[0],[1,0,0,0,0,0])))
print("Result CS : " + str(percentForMajor(df,'CS',[1,0,0,0,0,0])))

print(x[['Y', 'Class']])
#["Coding", "Artificial Intelligence", "Web", "Programming", "Business", "Engineer"]
data_Test = [0.5,0.5,0,0.5,0,0]
# Print Cluster
print(model.predict([data_Test]))

columns = ["Coding", "Artificial Intelligence",
           "Web", "Programming", "Business", "Engineer","Y"]

r = 6
listSubject = [0, 0, 0, 0, 0, 0]
report_df = pd.DataFrame(columns=columns)

numloc = 0

# for i in range(0, r):
#     listSubject[i] = 1
#     predict_Y = model.predict([listSubject])
#     tempList = listSubject
#     tempList.append(predict_Y[0])
#     report_df.loc[numloc] = tempList
#     numloc = numloc + 1
#     listSubject[i] = 0
#     del tempList[-1]

# for i in range(r-1):
#     for j in range(i, r-1):
#         listSubject[i] = 1
#         listSubject[j+1] = 1
#         predict_Y = model.predict([listSubject])
#         tempList = listSubject
#         tempList.append(predict_Y[0])
#         report_df.loc[numloc] = tempList
#         numloc = numloc + 1
#         listSubject[i] = 0
#         listSubject[j+1] = 0
#         del tempList[-1]
        
# z = pd.DataFrame()
# z = x[["Y", "Class"]]
# report_df = report_df.merge(z, how="inner")

# print(report_df)

# writer = pd.ExcelWriter('Report_Aptitute.xlsx', engine='xlsxwriter')
# report_df.to_excel(writer, sheet_name='result_Aptitute', index=False)
# writer.save()
