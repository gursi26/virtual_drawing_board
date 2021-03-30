import pandas as pd 

df1 = pd.read_csv('draw.csv')
df2 = pd.read_csv('erase.csv')
df3 = pd.read_csv('none.csv')

df4 = df1.append(df2)
df5 = df4.append(df3)
df5.to_csv("dataset.csv")