import knowledge_data
import pandas as pd

class SkillMajor:
    def __init__(self,dataUser):
        self.skill_Request = self.skill_request_main(dataUser)
        self.skill_Percentage = self.dataPercentTage(dataUser) 

    def skill_request_main(self,dataUser):
        subject_request = []
        subject_request.append(['CS',self.km_Skill_request(dataUser,'CS')])
        subject_request.append(['IT',self.km_Skill_request(dataUser,'IT')])
        subject_request.append(['CE',self.km_Skill_request(dataUser,'CE')])
        subject_request.append(['SE',self.km_Skill_request(dataUser,'SE')])
        subject_request.append(['BC',self.km_Skill_request(dataUser,'BC')])
        return subject_request

    def km_Skill_request(self,dataUser,req):
        df = knowledge_data.Skill().df
        
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

    def dataPercentTage(self,userData):
        df = knowledge_data.Skill().df
        percentTage = []
        percentTage.append(['CS',self.calculateSkillPercentage(df[(df['Y'] == 'CS')],userData)])
        percentTage.append(['IT',self.calculateSkillPercentage(df[(df['Y'] == 'IT')],userData)])
        percentTage.append(['CE',self.calculateSkillPercentage(df[(df['Y'] == 'CE')],userData)])
        percentTage.append(['SE',self.calculateSkillPercentage(df[(df['Y'] == 'SE')],userData)])
        percentTage.append(['BC',self.calculateSkillPercentage(df[(df['Y'] == 'BC')],userData)])
        
        return percentTage
        
    def calculateSkillPercentage(self,dataModel:pd.DataFrame,userData):
        sum =[]
        dataModel = dataModel.drop(columns='Y')
        data = dataModel.values.tolist()[0]
        
        for i in range(0,len(data)):
            sum.append(float(userData[i]) - float(data[i]) if float(data[i]) - float(userData[i]) > 0 else 0)
        
        return round((sum.count(0))/len(data),2)
            

    


v = SkillMajor([1,1,1,0,0.0,0])
print(v.skill_Request[0])
print(v.skill_Percentage[0])
print(v.skill_Request[1])
print(v.skill_Percentage[1])
print(v.skill_Request[2])
print(v.skill_Percentage[2])
print(v.skill_Request[3])
print(v.skill_Percentage[3])
print(v.skill_Request[4])
print(v.skill_Percentage[4])
