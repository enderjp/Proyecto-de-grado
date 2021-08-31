# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 12:20:07 2021

@author: Ender
"""

""" 
Script para concatenar vectores LBP con los distintos radios 

R = 1,2,4,6,8

"""


import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  f_classif
from sklearn.feature_selection import VarianceThreshold

print("Importando data...")

data1= pd.read_csv("utk_dataset/LBP_bloques96x96_1.csv",header=0)
data2= pd.read_csv("utk_dataset/LBP_bloques96x96_2.csv",header=0)
data3= pd.read_csv("utk_dataset/LBP_bloques96x96_4.csv",header=0)
data4= pd.read_csv("utk_dataset/LBP_bloques96x96_6.csv",header=0)
data5= pd.read_csv("utk_dataset/LBP_bloques96x96_8.csv",header=0)


#Feature selection
Fs= SelectKBest(score_func=f_classif)

#Lista con las caracteristicas de R=1...8
d= [data1,data2,data3,data4,data5]

# Extraer TODAS LAS etiquetas
#UTK
labels= data1[['genero','edad','raza']]
constant_filter = VarianceThreshold(threshold=0) # eliminr features con
                                                #  varianza cero


X_new=[]
for i in range(0,5):
    
    X= d[i].drop(['genero','edad','raza'],axis=1)
    #X= data.drop(['genero','raza','edad'],axis=1)
    
    # Remover características con varianza cero
    X=constant_filter.fit_transform(X)
   
   # k=int(len(X[1])/2)
   # x = SelectKBest(f_classif, k=k).fit_transform(X, Y)
    x=pd.DataFrame(X)
    X_new.append(x)
    print("Data %d procesada"  % i)
    print("Features:", X_new[i].shape)
    
print("Guardando features en un csv...")
Features_final=pd.concat([X_new[0],X_new[1],X_new[2],X_new[3],X_new[4],labels],axis=1)
Features_final.to_csv('LBP_utk_96x96_conc.csv',index=False)
# 





