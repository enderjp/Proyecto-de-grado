# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:06:18 2021

@author: Ender
"""

import pandas as pd
import numpy as np                                             

#----DATA--------#
data= pd.read_csv(" ",header=0)

#----------------------------
# caracteristicas


X= data
# etiquetas

labels= pd.read_csv(" ",header=0)


# Usar label encoder en caso que la etiqueta no esté codificada
# esto fue para probar el fair_face dataset, utk face dataset 
# ya viene codificado

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

Y=le.fit_transform(labels['gender'])

for i in range(len(labels)):
    if ((labels.loc[i,'gender'] =='Male')):
        labels.loc[i,'gender']=0
    else:
        labels.loc[i,'gender']=1

Y=labels['gender']
Y=np.ravel(Y)

#-------------------------------

# Más librerías
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time
#from joblibspark import register_spark
#from sklearn.utils import parallel_backend
from sklearn.preprocessing import  PowerTransformer

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


#register_spark() 

##-------Parámetros para el entrenamiento----#

# Validación cruzada de 5 carpetas
# usar una semilla para reproductividad
cv=StratifiedKFold(5, shuffle=True, random_state=200)


#-----Estimador/modelo-----#

# GENERO
#model=LogisticRegression( C=1,dual=False)
#model = LinearSVC(penalty='l2',verbose=4, dual=False)
model = SVC(kernel='rbf',verbose=4)  
#------------------------------------------------------------- ##


# Selección de características
fs = SelectKBest(score_func=f_classif) # ANOVA o f-test

# Método de aproximación de Kernel Nystroem
k=int(len(X[1])/4) # número de características escogido para aproxar el kernel
#Ny = Nystroem(kernel='rbf',random_state=1,n_components=k)

#----Definir el papeline a evaluar-------------#
# feature extraction + transformador  + training
pipeline = Pipeline(steps=[('anova',fs),
                           ('transformer',PowerTransformer()),
                          # ('nystroem',Ny), 
                           ('estimador', model)])

# Parámetros a evaluar
parameters = { 'anova__k':[k],
              #'nystroem__gamma':[0.00005],
              'estimador__C':[1]}


#Búsqueda de parámetros
search = GridSearchCV(pipeline, param_grid=parameters,cv=cv,scoring='accuracy', verbose=10)

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
                               target_names=['Masculino',
                                             'Femenino'],
                               output_dict=False)

#df = pd.DataFrame(summary).transpose()
#df.to_csv('classif_report_1.csv',index=False)

# PARA LFW
#summary= classification_report(y_test, y_pred, target_names=['Femenino','Masculino'])
print(summary)




# Graficar AUC


from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt


svc_disp = plot_roc_curve(search.best_estimator_, X_test, y_test)
#rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=svc_disp.ax_)

plt.show()
