from cProfile import label
from email.policy import default
import string
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
from sympy import Intersection

colum = ['สถิติ', 'การแจกแจงความน่าจะเป็นเบื้องต้น', 'ลำดับและอนุกรม',
    'แคลคูลัส', 'เรขาคณิตวิเคราะห์', 'เซต', 'ตรรกศาสตร์',
    'จำนวนจริงและพหุนาม', 'ฟังก์ชัน', 'ฟังก์ชันตรีโกณมิติ', 'จำนวนเชิงซ้อน',
    'เมทริกซ์', 'เวกเตอร์ในสามมิติ', 'หลักการนับเบื้องต้น', 'ความน่าจะเป็น',
    'ฟิสิกส์', 'เคมี', 'ชีวะ']

columnY = ["CS", "CE", "SE", "IT", "BC","CS", "CE", "SE", "IT", "BC","CS", "CE", "SE", "IT", "BC"]


def stringToList(data:string):
    result =[]
    txt = ""
    for i in data:
        txt = txt + i
        if i == ',':
            txt = txt.strip(',')
            number = float(txt)
            result.append(number)
            txt = ""

    txt = txt.strip(',')
    number = float(txt)
    result.append(number)

    return result

def writeFullName(majorName):
    if majorName == "CS":
        return "Computer Science"
    elif majorName == "CE":
        return "Computer Engineering"
    elif majorName == "SE":
        return "Software Engineering"
    elif majorName == "IT":
        return "Information Technology"
    else:
        return "Business Computer"

def km_Subject(dataTestUser):
    df = pd.read_excel('DatabaseOK.xlsx',sheet_name="DB")

    # แปลง String to List
    marterial = stringToList(dataTestUser)
    
    x = df[colum]
    
    req = "BC"

    model = KMeans(n_clusters=5)
    y_Kmeans = model.fit_predict(x.values)

    z = x
    z["Y"] = y_Kmeans
    z['Class'] = columnY

    # print(z[['Y','Class']])

    data_Test =[0	,0	,0	,0	,0	,0	,0	,0	,0	,0	,0	,0	,0	,0	,0	,1	,1	,1]

    # print(data_Test)
    # result = model.predict([data_Test])
    # print(result)

    predicValue = model.predict([data_Test])[0]
    resultUser = model.predict([marterial])[0]

    # print()
    v = z["Class"].loc[z['Y']== predicValue]
    
    resultName = writeFullName(z["Class"].loc[z['Y']== resultUser].values[0])
    
    # print("Result : " + v.values[0])
    # print()

    #นำ Columns เข้าวิชาที่ต้องเรียน intersection กับ โครงสร้างคณะที่เลือก
    rowRequest = df.loc[x["Class"] == req]
    # print("Row Request")
    # print(rowRequest)

    rowRequest = rowRequest.drop(["Y"], axis=1)
    columnRequest = rowRequest.columns.values
    classRequest = rowRequest.values.tolist()[0]

    resultRequest = []

    #วิชาที่ต้องเรียน
    # print(classRequest)

    #วิชาที่รับเข้า
    # print(data_Test)
    # print()

    #ค้นหาวิชาที่ต้องเรียนเพิ่ม
    for i in range(0,len(marterial)):
        resultRequest.append(float(marterial[i]) - float(classRequest[i]))

    reqSubject = list()

    for i in range(0,len(columnRequest)):
        if(resultRequest[i] < 0):
            # print(columnRequest[i] + " " + str(resultRequest[i]) + " Not PASS")
            reqSubject.append(columnRequest[i])
        # else:
        #     print(columnRequest[i] + " " + str(resultRequest[i]) + " PASS")

    # print()
    # print("Required subjects " + str(reqSubject[:]))
    return resultName,reqSubject


dataGroupJob = {
    0:["Programmer","Developer","Business Analyst"],
    1:["Cloud & Infrastructure","Application Analyst"],
    2:["Software Engineer","Data Scientist"],
    3:["Data Engineer"],
    4:["AI / Machine Learning Engineer"]
}

dataGroup ={
    0:["Computer Science","Computer Engineering","Software Engineering","Information Technology","Business Computer"],
    1:["Computer Science","Computer Engineering","Software Engineering","Information Technology"],
    2:["Computer Science","Computer Engineering","Software Engineering"],
    3:["Computer Engineering","Software Engineering"],
    4:["Computer Science","Computer Engineering"]
}

# Strip String Job user require
def splitJob(dataReq:string):
    result = []
    word =""
    for i in dataReq:
        if i == "," or i == len(dataReq):
            txt = word.strip(',')
            result.append(txt)
            word =""
        word = word + i

    txt = word.strip(',')
    result.append(txt)

    return result

def km_Subject_Req(dataSubject,dataReq):
    resultSubject,reqSubject = km_Subject(dataSubject)
    resultRequire = []
    jobRequire = splitJob(dataReq)
    
    checkNumGroup = []

    for i in range(0,len(dataGroup)):
        for j in dataGroup[i]:
            if resultSubject == j:
                checkNumGroup.append(i)

    job = []  
    for i in range(0,len(checkNumGroup)):
        for j in dataGroupJob[i]:
            job.append(j)
    
    resultRequire = intersection(job,jobRequire)

    return resultSubject,reqSubject,resultRequire

def intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def km_JobReq(major,dataReq):
    resultSubject = major
    resultRequire = []
    jobRequire = splitJob(dataReq)
    
    checkNumGroup = []

    for i in range(0,len(dataGroup)):
        for j in dataGroup[i]:
            if resultSubject == j:
                checkNumGroup.append(i)
    print(checkNumGroup)
    job = []  
    for i in range(0,len(checkNumGroup)-1):
        for j in dataGroupJob[i]:
            job.append(j)
            
    print(job)

    resultRequire = intersection(job,jobRequire)
    if resultRequire == []:
        resultRequire = "Empty"

    return resultRequire