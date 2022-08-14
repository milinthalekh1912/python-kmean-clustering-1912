import pandas as pd
from pyexpat import model
from sklearn.cluster import KMeans

df = pd.read_excel("DatabaseOK.xlsx",sheet_name="SKILLFORJOB")

#0 Programer,Developer = Coding ,Programming ,Project = https://www.indeed.com/career-advice/finding-a-job/developer-vs-programmer
#1 Business Analyst,Application Analyst = Business ,Coding ,Programming ,Project = https://www.cio.com/article/276798/project-management-what-do-business-analysts-actually-do-for-software-implementation-projects.html
# Application Analyst = https://www.fieldengineer.com/skills/application-analyst
#2 Cloud & Infrastructure = Coding ,Programming ,Project ,Artificial Intelligence = https://www.techtarget.com/whatis/feature/Top-20-cloud-computing-skills-to-boost-your-career
#4 Software Engineer = Coding ,Programming ,Project ,Engineer
#  https://www.monster.com/career-advice/article/software-engineer-skills

#5 Data Scientist = Coding ,Programming ,Project ,Artificial Intelligence
# = https://www.simplilearn.com/what-skills-do-i-need-to-become-a-data-scientist-article
#6 Data Engineer = Coding ,Programming ,Project ,Artificial Intelligence ,Engineer
#  = https://www.indeed.com/career-advice/resumes-cover-letters/data-engineer-skills
#7 AI / Machine Learning Engineer = Coding ,Programming ,Project ,Engineer ,Artificial Intelligence
#  = https://www.geeksforgeeks.org/7-skills-needed-to-become-a-machine-learning-engineer/


dataRequestGroup = {
    0: ["Coding", "Programming", "Project"],
    1: ["Business", "Coding", "Programming", "Project"],
    2: ["Coding", "Programming", "Project", "Artificial Intelligence"],
    3: ["Coding", "Programming", "Project", "Engineer"],
    4: ["Coding", "Programming", "Project", "Artificial Intelligence"],
    5: ["Coding", "Programming", "Project", "Artificial Intelligence", "Engineer"],
}

dataGroupJob = {
    0:["Programmer","Developer","Business Analyst"],
    1:["Cloud & Infrastructure","Application Analyst"],
    2:["Software Engineer","Data Scientist"],
    3:["Data Engineer"],
    4:["AI / Machine Learning Engineer"]
}
def clusteringSkill(data,jobRequire):
    df = pd.read_excel("DatabaseOK.xlsx", sheet_name="SKILLFORJOB")
    x = df.drop("Y", axis=1)

    model = KMeans(n_clusters=9)
    y_Kmeans = model.fit_predict(x)
    x['Y'] = df['Y']
    x['Cluster'] = y_Kmeans
    print(x[['Y', 'Cluster']])

    predicValue = model.predict([data])[0]

    print()
    rowRequest = df.loc[x["Y"] == jobRequire]

    columnRequest = rowRequest.columns.values
    classRequest = rowRequest.values.tolist()[0]

    resultRequest = []

    for i in range(0, len(data)):
        resultRequest.append(float(data[i]) - float(classRequest[i]))

    for i in range(0, len(columnRequest)-1):
        if(resultRequest[i] < 0):
            print(columnRequest[i] + " Not PASS")
        else:
            print(columnRequest[i] + " PASS")
            
print(clusteringSkill([1, 1, 1, 0, 0, 0],"Programmer"))