# -*- coding: utf-8 -*-
"""concatenar_features.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xepjLtpuz59Z8n_4nNTHwNVATupfRIdD
"""

#----------------------------------------------------------
# Concatenacion de feature vectors a diferentes escalas
#-----------------------------------------------------------
#Librerías
import numpy as np 
import pandas as pd


# # Importar las caracteristicas y dejar solo el genero como etiqueta en el último csv a unir
datos1=pd.read_csv(" ",header=0)
#datos1=datos1.drop(columns=['genero','edad','raza'])

datos2=pd.read_csv(" ",header=0)
#datos2=datos2.drop(columns=['edad','raza'])


data=pd.concat([datos1,datos2], axis=1, ignore_index=True)

#index_ed=len(data)-1

# Renombrar las ultimas 3 columnas como corresponde (features)


data.rename(columns = {len(data.columns)-1:'edad'}, inplace = True)

# UTK dataset

# # Importar las caracteristicas y dejar solo el genero como etiqueta en el último csv a unir
# datos1=pd.read_csv("lfw_dataset/HOG_lfw_8_32x32.csv",header=0)
# datos1=datos1.drop(columns=['genero'])

# datos2=pd.read_csv("lfw_dataset/HOG_lfw_8_64x64.csv",header=0)
# datos2=datos2.drop(columns=['genero'])

# datos3=pd.read_csv("lfw_dataset/HOG_lfw_8_96x96.csv",header=0)
# #datos3=datos3.drop(columns=['genero'])

# data=pd.concat([datos1,datos2,datos3], axis=1, ignore_index=True)


# index_ed=len(data)-1

# Renombrar las ultimas 3 columnas como corresponde (features)

# data.rename(columns = {len(data.columns)-3:'genero'}, inplace = True) 
# data.rename(columns = {len(data.columns)-2:'edad'}, inplace = True) 
# data.rename(columns = {len(data.columns)-1:'raza'}, inplace = True) 

#data.rename(columns = {len(data.columns)-1:'genero'}, inplace = True)

# verificar si hay filas que contengan valores nulos
nan_rows = data[data.isnull().any(1)]
print(nan_rows)

# data=data.drop([6156,10840,17878,21413,22579],axis=0)

# Se eliminan dichas filas

data= data.dropna(how='any')

# necesario hacer unpack labels con **
#datos= datos1.assign(**datos2)

## Guardar características y etiquetas en un csv
data.to_csv('features_final.csv',index=False)
