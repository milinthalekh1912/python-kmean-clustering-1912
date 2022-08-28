import knowledge_data
from sklearn.cluster import KMeans
import pandas as pd

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

class PredictData:
    def __init__(self,dataUser):
        self.major_percentage = self.dataPercentTage(self.predit_fit(dataUser))
        self.subject_request_data = self.subject_request_main(dataUser)
        self.major_job_result = self.major_job_request()

    def dataPercentTage(self,dataModel):
        percentTage = []
        percentTage.append(['CS',self.calculatePercentage(dataModel,'CS')])
        percentTage.append(['IT',self.calculatePercentage(dataModel,'IT')])
        percentTage.append(['CE',self.calculatePercentage(dataModel,'CE')])
        percentTage.append(['SE',self.calculatePercentage(dataModel,'SE')])
        percentTage.append(['BC',self.calculatePercentage(dataModel,'BC')])
        return percentTage
        
    def calculatePercentage(self,dataModel,majorName):
        return len(dataModel[(dataModel['Y'] == majorName)])/len(dataModel)

    def predit_fit(self,dataUser):
        df = knowledge_data.KnowLedge('dataservey').df
        table = df.drop(columns='Y')
        model = KMeans(n_clusters=5)
        y_Kmeans = model.fit_predict(table)
        table['Cluster'] = y_Kmeans
        table['Y'] = df['Y']
        
        result = self.dropClusterTable(model.predict([dataUser])[0],table)
        return result
        
        
    def dropClusterTable(self,cluster,tableData):
        return tableData.drop(tableData[tableData['Cluster'] != cluster].index)

    def subject_request_main(self,dataUser):
        subject_request = []
        subject_request.append(['CS',self.km_Subject_request(dataUser,'CS')])
        subject_request.append(['IT',self.km_Subject_request(dataUser,'IT')])
        subject_request.append(['CE',self.km_Subject_request(dataUser,'CE')])
        subject_request.append(['SE',self.km_Subject_request(dataUser,'SE')])
        subject_request.append(['BC',self.km_Subject_request(dataUser,'BC')])
        return subject_request


    def km_Subject_request(self,dataUser,req):
        df = knowledge_data.KnowLedge('knowledge_request').df
        
        rowRequest = df.loc[df["Y"] == req]

        rowRequest = rowRequest.drop(["Y"], axis=1)
        columnRequest = rowRequest.columns.values
        classRequest = rowRequest.values.tolist()[0]

        resultRequest = []

        for i in range(0,len(dataUser)):
            resultRequest.append(float(dataUser[i]) - float(classRequest[i]))

        reqSubject = list()

        for i in range(0,len(columnRequest)):
            if(resultRequest[i] < 0):
                reqSubject.append(columnRequest[i])
            
        return reqSubject
    
    def major_job_request(self):
        major_job = []
        major_job.append(['CS',self.km_JobReq('Computer Science')])
        major_job.append(['CE',self.km_JobReq('Computer Engineering')])
        major_job.append(['SE',self.km_JobReq('Software Engineering')])
        major_job.append(['IT',self.km_JobReq('Information Technology')])
        major_job.append(['BC',self.km_JobReq('Business Computer')])
        return major_job

    def km_JobReq(self,major):
        checkNumGroup = []

        for i in range(0,len(dataGroup)):
            for j in dataGroup[i]:
                if major == j:
                    checkNumGroup.append(i)
        
        job = []  
        for i in range(0,len(checkNumGroup)):
            for j in dataGroupJob[i]:
                job.append(j)
    
        return job


v = PredictData([0.2,0.333333333,0,0,0,1,0,0.285714286,0.285714286,0.5,0,0,0,0.25,0,0.388888889,0.25,0.4375])
print(v.subject_request_data[0])
print(v.major_percentage[0])

print("================================")
print(v.subject_request_data[1])
print(v.major_percentage[1])

print("================================")
print(v.subject_request_data[2])
print(v.major_percentage[2])

print("================================")
print(v.subject_request_data[3])
print(v.major_percentage[3])

print("================================")
print(v.subject_request_data[4])
print(v.major_percentage[4])

print(v.major_job_result)