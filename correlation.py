from tkinter import font
from cv2 import rotate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_excel("Attribute.xlsx",sheet_name="HeadOfSub")
plt.figure(figsize=(6,4))
cor = df.corr()
g = sns.heatmap(cor,annot=True,cmap='YlGnBu',linewidths=1,linecolor='b',fmt='.2f')
g.set_yticklabels(g.get_yticklabels(),rotation=0,fontsize=1)
plt.show()
