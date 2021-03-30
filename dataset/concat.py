import pandas as pd 

df1 = pd.read_csv('draw.csv')
df2 = pd.read_csv('erase.csv')

df3 = df1.append(df2)
df3.to_csv("dataset.csv")