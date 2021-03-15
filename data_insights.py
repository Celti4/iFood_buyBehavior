# -*- coding: utf

# LIBRARY IMPORT
#from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy as sp
import random
import statistics

random.seed(174853)

# DATA MANIPULATION
df = pd.read_csv(r'D:\Projetos\iFood\dadosIgnacio.csv', delimiter = ";", parse_dates= ['data_pedido'], dayfirst= True)

df["Year-Month"] = df['data_pedido'].dt.strftime('%Y-%m')
df["Year"] = df['data_pedido'].dt.strftime('%Y')
df["Pandemia"] = np.where(df["Year-Month"] < "2020-03", "Normal", "Covid")


df.groupby(['Year-Month']).valor.agg(['sum', 'max','mean', 'count'])

df.groupby(['Year']).valor.agg(['sum', 'max', 'mean', 'count'])

df.groupby(['Pandemia']).valor.agg(['sum', 'max', 'mean', 'count'])

dfMes = df.groupby(['Year-Month', "Pandemia"]).valor.agg(['sum', 'max','mean', 'count']).reset_index("Pandemia")
dfMes = dfMes.filter(["Year-Month", "mean", "Pandemia", "count", "sum"])
dfMes = dfMes.drop("2021-02")
dfMes.index = pd.to_datetime(dfMes.index)
dfMes = dfMes.rename(columns = {"mean" : "Average Spend", "count" : "Orders", "sum" : "Total"})


isNormal = dfMes.index <= "2020-03"
isCovid = dfMes.index >= "2020-03"


dfMesNormal = dfMes[isNormal]
dfMesCovid = dfMes[isCovid]
Media_Covid = df.groupby("Pandemia").valor.agg(['mean']).loc["Covid", "mean"]
Media_Normal = df.groupby("Pandemia").valor.agg(['mean']).loc["Normal", "mean"]
Media_Covid_Frequencia = dfMes.groupby("Pandemia").Orders.agg(['mean']).loc["Covid", "mean"]
Media_Normal_Frequencia = dfMes.groupby("Pandemia").Orders.agg(['mean']).loc["Normal", "mean"]
dfMesNormal["Mean"] = Media_Normal
dfMesCovid["Mean"] = Media_Covid
dfMesNormal["AverageOrders"] = Media_Normal_Frequencia
dfMesCovid["AverageOrders"] = Media_Covid_Frequencia




#TESTE DE WILCOXON-MANN-WHITNEY
#Valor
index_Normal = df.Pandemia == "Normal"
index_Covid = df.Pandemia == "Covid"
X_Normal = df.valor[index_Normal]
Y_Covid = df.valor[index_Covid]

#Frequencia
index_freqN = dfMes.index <= "2020-02"
index_freqC = dfMes.index > "2020-02"
X_FreqN = dfMes.Orders[index_freqN]
Y_FreqC = dfMes.Orders[index_freqC]

sp.stats.mannwhitneyu(X_Normal, Y_Covid, alternative = "less").pvalue 
sp.stats.mannwhitneyu(X_FreqN, Y_FreqC, alternative = "less").pvalue



### Modelo de NPD

df["weekDay"] = df['data_pedido'].dt.strftime('%a')
df.groupby(["weekDay", "Pandemia"]).valor.agg("count")

dfCovid = df[df.Pandemia == "Covid"].copy().reset_index(drop = True)
weekDay_prob = round(dfCovid.groupby(["weekDay"]).valor.count()*100/dfCovid.valor.count())
weekDay_prob = weekDay_prob.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

dfCovid['difTime'] = (dfCovid['data_pedido'][0] - dfCovid['data_pedido'][0])
dfCovid['difTime_Next'] = (dfCovid['data_pedido'][0] - dfCovid['data_pedido'][0])

for i in range(1,len(dfCovid)):
    dfCovid['difTime'][i] = (dfCovid["data_pedido"][i] - dfCovid['data_pedido'][i-1])
    dfCovid['difTime_Next'][i-1] = (dfCovid["data_pedido"][i] - dfCovid['data_pedido'][i-1])

dfCovid['difTime'] = dfCovid['difTime'].astype('timedelta64[D]')
dfCovid['difTime'][0] = 4
dfCovid['difTime_Next'] = dfCovid['difTime_Next'].astype('timedelta64[D]')



#indexTrain = np.random.rand(len(dfCovid)) < 0.75
indexTrain = dfCovid["Year-Month"] < "2020-12"

train_dfCovid = dfCovid[indexTrain]
test_dfCovid = dfCovid[~indexTrain]
test_dfCovid = test_dfCovid[test_dfCovid['difTime_Next'] > 0]


Pivot = train_dfCovid.groupby(["difTime_Next", "weekDay"]).id_usuario.count().reset_index()
Pivot.columns = ['difTime', "weekDay", "values"]
Pivot.head(5)
Pivot = Pivot[Pivot.difTime > 0]
Pivot = Pivot[Pivot.difTime < 8]
Pivot = pd.pivot_table(Pivot, values = ["values"], index = ["weekDay"], columns = ["difTime"], aggfunc = np.sum, fill_value = 0)
Pivot = Pivot.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
Pivot = Pivot.rename(columns = {"1.0" : "1D", "2.0" : "2D", "3.0" : "3D", "4.0" : "4D", "5.0" : "5D", "6.0" : "6D", "7.0": '7D'})

#Pivot.iloc[0,3] 



# Rearrange the columns row by row. 
Monday = np.concatenate(
    [Pivot.iloc[0,6:7].values, # Linha 1, Coluna 7
     Pivot.iloc[0, 0:6].values] # Linha 1 , Coluna 1:6
    )

Tuesday = np.concatenate(
    [Pivot.iloc[1,5:7].values, 
     Pivot.iloc[1, 0:5].values]
    )

Wednesday = np.concatenate(
    [Pivot.iloc[2,4:7].values, 
     Pivot.iloc[2, 0:4].values]
    )

Thursday = np.concatenate(
    [Pivot.iloc[3,3:7].values, 
     Pivot.iloc[3, 0:3].values]
   )

Friday = np.concatenate(
    [Pivot.iloc[4,2:7].values, 
     Pivot.iloc[4, 0:2].values]
    )

Saturday =np.concatenate(
    [Pivot.iloc[5,1:7].values, 
     Pivot.iloc[5, 0:1].values]
                         )

Sunday = np.concatenate(
    [Pivot.iloc[6,0:7].values, 
     Pivot.iloc[6, 0:0].values]
    )


Markov_chain = pd.DataFrame(np.vstack((Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)))

Markov_chain = Markov_chain.rename(index = {
        0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    )

Markov_chain = Markov_chain.rename(columns = {
        0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    )

Markov_chain_prob = Markov_chain.copy()

for i in range(0,len(Markov_chain)):
    Markov_chain_prob.iloc[i,:] = Markov_chain.iloc[i,:]/Markov_chain.iloc[i, :].sum()

Equal_distribution = np.array([100/7,100/7,100/7,100/7,100/7,100/7,100/7])

Result = pd.DataFrame(np.dot(Equal_distribution, Markov_chain_prob).round(decimals = 0))

Result = Result.rename(index = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"})



#### TEST RESULTS
# Trying to predict the next purchase date

from NextPurchaseDateMarkov import MarkovNextPurchaseDate

test_dfCovid = test_dfCovid[test_dfCovid["Year-Month"] == "2020-12"]
testDayList = pd.DataFrame(test_dfCovid['weekDay'].reset_index(drop = True))

resultsList = []

for j in range(0, 1000):
    count = 0 
    for i in range(len(testDayList)-1):
        if(MarkovNextPurchaseDate(testDayList.weekDay[i], Markov_chain_prob) == testDayList.weekDay[i+1]): 
            count = count +1
            
    resultsList = np.append(resultsList, count)
    
    

statistics.mean(resultsList)/(len(testDayList)-1) # Mean Accuracy

### Creating reference ###

dfGraphReference = pd.DataFrame(columns = ["Round", "modelScore", "randomScore", "perfectScore"])

from NextPurchaseDateMarkov import randomDay

scoreModel = 0
randomScore = 0 
for i in range(1, len(testDayList)): 
    if(MarkovNextPurchaseDate(testDayList.weekDay[i-1], Markov_chain_prob) == testDayList.weekDay[i]): 
        scoreModel = scoreModel +1
    
    if(randomDay() == testDayList.weekDay[i]):
        randomScore = randomScore +1
    
    aux = []
    aux.append([i, scoreModel, randomScore, i])
    aux = pd.DataFrame(aux, columns = ["Round", "modelScore", "randomScore", "perfectScore"])
    
    dfGraphReference = pd.concat([dfGraphReference, aux])









