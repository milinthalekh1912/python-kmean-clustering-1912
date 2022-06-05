import pandas as pd
from pyexpat import model
from sklearn.cluster import KMeans

df = pd.read_excel("DatabaseOK.xlsx",sheet_name="Skill")


x = df.drop("Y",axis=1)
req = "CE"

model = KMeans(n_clusters=5)
y_Kmeans = model.fit_predict(x.values)

z = x
z["Y"] = y_Kmeans
z['Class'] = ["CS", "CE", "SE", "IT", "BC"]

print(z[['Y','Class']])

data_Test = [1,1,1,0,0,0]

# result = model.predict([data_Test])
predicValue = model.predict([data_Test])[0]
print()
v = z["Class"].loc[z['Y']== predicValue]
print(v.values[0])
print()
rowRequest = df.loc[x["Class"] == req]
columnRequest = rowRequest.columns.values
classRequest = rowRequest.values.tolist()[0]

resultRequest = []
print()
for i in range(0,len(data_Test)):
    resultRequest.append(float(data_Test[i]) - float(classRequest[i]))

for i in range(0,len(columnRequest)-1):
    if(resultRequest[i] < 0):
        print(columnRequest[i] + " Not PASS")
    else:
        print(columnRequest[i] + " PASS")


