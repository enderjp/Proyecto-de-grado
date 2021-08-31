# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:03:28 2021

@author: Ender
"""


""" 
Script para el entrenamiento de la etiqueta edad

"""

#--------- En caso de usar spark --------------#
# import findspark 
# findspark.init("C:\spark\spark-3.0.1-bin-hadoop2.7") # spark 3.0 para usar 
# from joblibspark import register_spark
# from sklearn.utils import parallel_backend


# # Activar cluster spark para el entrenamiento
# register_spark() 

#----------------------------------#

import pandas as pd

data= pd.read_csv("utk_dataset/LBP_bloques32x32_2.csv",header=0)


#------Preprocesamiento---------#

# Balancear las clases
# Unir clases 3 y 4
#data=data['raza']
for i in range(len(data)):
    if data.loc[i,'raza']==4:
       data.loc[i,'raza']=3

 # Categorizar edades       
for i in range(len(data)):
    if ((data.loc[i,'edad'] >=0) & (data.loc[i,'edad'] <=10)):
        data.loc[i,'edad']=0
    elif ((data.loc[i,'edad'] >=11) & (data.loc[i,'edad'] <=20)):
        data.loc[i,'edad']=1
    elif ((data.loc[i,'edad'] >=21) & (data.loc[i,'edad'] <=35)):
        data.loc[i,'edad']=2
    elif ((data.loc[i,'edad'] >=36) & (data.loc[i,'edad'] <=50)):
        data.loc[i,'edad']=3 
    elif ((data.loc[i,'edad'] >=51) & (data.loc[i,'edad'] <=65)):
          data.loc[i,'edad']=4
    else:
        data.loc[i,'edad']=5


#---------------------------------#
    
# caracteristicas
X= data.drop(['edad','genero','raza'],axis=1)

# Etiquetas
#Y=labels
Y= data['edad']
#Y=np.ravel(Y)

#-------------------------------
# Más librerías
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif

# Remover características con varianza cero
from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0)
X=constant_filter.fit_transform(X)
#X.shape


print("Eliminando features con varianza cero, se obtiene ahora un vector de:", X.shape)
# Dividir la data de entrenamiento y validación
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=500,shuffle=True)


# Estandarización
from sklearn.preprocessing import PowerTransformer
#----------------------------------------




##-------Parámetros para el entrenamiento----#

# Validación cruzada de 5 carpetas
cv=StratifiedKFold(5, shuffle=True, random_state=10)


#------Definir el pipeline del modelo---------------#

#-----Estimador/clasificador-----#
#model = LinearSVC(verbose=4, dual=False,max_iter=3000)
#                     class_weight='balanced')
                   #  multi_class='ovr')


model = SVC(kernel='rbf',verbose=4) 


#---- Selección de características----#
fs = SelectKBest(score_func=f_classif) # ANOVA o f-test

k=int(X.shape[1]/4) # número de características escogido 
                    # con ANOVA

#------Pipeline ------#
pipeline = Pipeline(steps=[('anova',fs),
                           ('scale',PowerTransformer()),
                            ('estimador', model)])

# Parámetros a evaluar
parameters = { 'anova__k':[k],
              'estimador__C':[1]}


#-------- Búsqueda de parámetros ---------#

search = GridSearchCV(pipeline, param_grid=parameters,cv=cv,scoring='accuracy', verbose=4)

tiempo_inicial=time()

#with parallel_backend('spark'): # Utilizar el cluster de spark
search.fit(X_train, y_train)
    
tiempo_final=time()    

#Mostrar resultados
print('Best Mean Accuracy-score: %.3f' % search.best_score_)
print('Best Config: %s' % search.best_params_)

tiempo_ejecucion = tiempo_final - tiempo_inicial
print('Tiempo de ejecución', tiempo_ejecucion)
resumen= pd.DataFrame(search.cv_results_)

# Guardar resumen
resumen.to_csv('edad1.csv',index=False)


#-------------------------------------------

# Matriz de confusión
y_pred=search.best_estimator_.predict(X_test)
from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test, y_pred))


from sklearn.metrics import classification_report
# PARA UTK
summary= classification_report(y_test, y_pred, 
                               target_names=['0-10','11-20','21-35',
                                             '36-50',
                                             '51-65','66+'])

print(summary)

