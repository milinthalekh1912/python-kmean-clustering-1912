from sklearn.cluster import KMeans
from nbformat import read
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("Attribute.xlsx",sheet_name="HeadOfSub")

pca = PCA()  
sc = StandardScaler()

X = df.drop('Class',axis=1)
cols = X.columns
cols = cols.to_list()

X_sc = sc.fit_transform(X)

sc_df = pd.DataFrame(X_sc,columns=cols)

# writer = pd.ExcelWriter('pcaAttribute.xlsx', engine='xlsxwriter')
# sc_df.to_excel(writer, sheet_name='Sheet1', index=False)
# writer.save()

X_pca = pca.fit_transform(X_sc)
# print(X.describe().round(2))
# print(sc_df.describe().round(2))

pca_df = pd.DataFrame(X_pca)

# writer = pd.ExcelWriter('pcaAttributeDf.xlsx', engine='xlsxwriter')
# pca_df.to_excel(writer, sheet_name='Sheet1', index=False)
# writer.save()

# ------------------------------------------------------------------------------
model = KMeans(n_clusters=19)
y_Kmeans = model.fit_predict(X)


dataTest = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0.5, 0.5, 1, 1, 1, 0.5]

result =model.predict([dataTest])
X["Cluster"] = y_Kmeans
print(X)
print(dataTest)
print(result)
