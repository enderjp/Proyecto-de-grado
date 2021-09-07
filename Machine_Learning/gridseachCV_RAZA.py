# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:39:41 2021

@author: Ender
"""


""" 
Script para el entrenamiento de la etiqueta raza

"""
#---------Spark --------------#
#import findspark 
#findspark.init("C:\spark\spark-3.0.1-bin-hadoop2.7") # spark 3.0
#register_spark() 


import pandas as pd
import numpy as np     
                                        
#----DATA--------#
data= pd.read_csv(" ",header=0)

#-----------------------------------------------------#
# ---------------- Preprocesamiento -----------------#
# Unir clases 3 y 4, para tener clases más balancedas
# La clase 4 se adhiere a la 3, quedan 4 clases en total.
print("Uniendo las clases 3 y 4...")
for i in range(len(data)):
    if data.loc[i,'raza']==4:
        data.loc[i,'raza']=3
raza=data['raza']        
print("Cantidad de muestras por clase:")        
print(raza.value_counts())   
#----------------------------------
# Se eliminan aleatoriamente algunas muestras de la clase 0
# ya que hay muchas más respecto a las demás y esto genera desbalanceo
# de clases, lo que hace más difícil para el algoritmo 
# aprender bien las clases minoritarias
#



print("Terminado. Ahora las clases están más balanceadas.")

print("Cantidad de muestras por clase:")     
raza=data['raza']
print(raza.value_counts())
#---------------------------------------


#----------------------------
# caracteristicas


X= data.drop(['genero','raza','edad'],axis=1)
# etiquetas
#Y=data['raza']
Y=data['raza']
#Y= data[['edad']].values
Y=np.ravel(Y)

#-------------------------------

# Más librerías
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time
#from joblibspark import register_spark
#from sklearn.utils import parallel_backend
from sklearn.preprocessing import PowerTransformer

#------------------------------------------------

# Remover características con varianza cero
from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0)
X=constant_filter.fit_transform(X)
#X.shape
print("Eliminando features con varianza cero, se obtiene ahora un vector de:", X.shape)
#-------------------------------------------------------#

#------Dividir la data de entrenamiento y validación------#

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=200,shuffle=True)




##-------Parámetros para el entrenamiento----#

# Validación cruzada de 5 carpetas
cv=StratifiedKFold(5, shuffle=True, random_state=200)


#-----Estimador/clasificador-----#

# GENERO
#model=LogisticRegression( C=1,dual=False)
#model = LinearSVC(C=1,penalty='l2',verbose=4, dual=False)
#model = SVC(kernel='rbf',verbose=4,decision_function_shape='ovr')  
#------------------------------------------------------------- ##

# RAZA

# model = LinearSVC(verbose=4, dual=False,
#                     # class_weight='balanced',
#                       multi_class='ovr')

model = SVC(kernel='rbf',verbose=4) 

#--------------------------------------------------

# Selección de características
fs = SelectKBest(score_func=f_classif) # ANOVA o f-test

# Método de aproximación de Kernel Nystroem
k=int(len(X[1])/4) # número de características escogido para aproximar el kernel


#----Definir el papeline a evaluar-------------#
# feature extraction + kernel aproximation + training
pipeline = Pipeline(steps=[('anova',fs),
                           ('transformer',PowerTransformer()),
                           ('estimador', model)],verbose=3)

# Parámetros a evaluar
parameters = { 'anova__k':[k],
              #'nystroem__gamma':[0.00005],
              'estimador__C':[1]}


#Búsqueda de parámetros
search = GridSearchCV(pipeline, param_grid=parameters,cv=cv,scoring='accuracy', verbose=4)

tiempo_inicial=time()
#with parallel_backend('spark'):
search.fit(X_train, y_train)
    
tiempo_final=time()    
#X_new = search.fit_transform(X)
#Mostrar resultados
print('Best Mean Accuracy-score: %.3f' % search.best_score_)
print('Best Config: %s' % search.best_params_)

tiempo_ejecucion = tiempo_final - tiempo_inicial
print('Tiempo de ejecución', tiempo_ejecucion)
resumen= pd.DataFrame(search.cv_results_)

# Guardar resumen
resumen.to_csv('HOG16_1.csv',index=False)

# Matriz de confusión
y_pred=search.best_estimator_.predict(X_test)
from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test, y_pred))

#----------------------------
# cont=0
# for i in range(0,len(y_test)):
#     if y_test[i]==1:
#         cont+=1

# print(cont)
#----------------------

#Import scikit-learn metrics module for accuracy calculation


# Model Accuracy: how often is the classifier correct?
#print("Accuracy:",metrics.balanced_accuracy(y_test, y_pred))

from sklearn.metrics import classification_report
# PARA UTK
summary= classification_report(y_test, y_pred, 
                               target_names=['Caucásico',
                                             'Africano/Afroamericano',
                                             'Asiático',
                                            'Latino/Med.Oriente/Indio'],
                               output_dict=False)

#df = pd.DataFrame(summary).transpose()
#df.to_csv('classif_report_1.csv',index=False)

print(summary)




# Graficar AUC


# from sklearn.metrics import plot_roc_curve
# import matplotlib.pyplot as plt


# svc_disp = plot_roc_curve(search.best_estimator_, X_test, y_test)
# #rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=svc_disp.ax_)

# plt.show()


