# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 21:41:30 2020

@author: Ender
"""

#----------------------------------------#
##------ Feature extraction - LBP------- ##
#----------------------------------------#

import cv2 
import os

## --------------------funciones

# variable para almacenar los nombres de las imagenes, a fin de mantenerlos
# iguales luego de alinearlas y recortarlas
files_names=[]

# # Cargar imagenes desde una carpeta
# def cargar_imagenes(path):
       
#        images = []
#        for filename in os.listdir(path):
           
         
#            img = cv2.imread(os.path.join(path,filename))
           
#            files_names.append(filename)
#            if img is not None:
#                gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                images.append(gray)
               
#        return images
   
def cargar_imagenes(path):
       images = []
       for  root, dirs,filename in os.walk(path):
           for file in filename:
               img = cv2.imread(os.path.join(root,file))
               files_names.append(file)
               if img is not None:
                   gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                   images.append(gray)
       return images    


##--------------------------------------
images = []

# --- Comentar o descomentar según el dataset
path="fair_face/fair_face_96x96/"
#path1="imagenes_resized_64x64/female/"
#path2= "imagenes_resized_64x64/male/"
#path="UTKFace/"

# Unir ambos generos, se hace así para conservar el orden de las etiquetas

print("Importando imágenes...")

#Imagenes1= cargar_imagenes(path1) # female
#Imagenes2 = cargar_imagenes(path2) # male
#Imagenes = Imagenes1+Imagenes2
Imagenes= cargar_imagenes(path)

print("Calculando LBP...")
##---------- Local Binary Pattern ----------##

from skimage.feature import local_binary_pattern

import numpy as np 
import pandas as pd

radius= 8#8
P=8#16 #vecindades
Lbp = []
M = Imagenes[0].shape[0]//4
N = Imagenes[0].shape[1]//4
#y=1
n=1
for img in Imagenes:
    print("procesando imagen n°:",n)
    # if y<=10:
    #     cv2.imshow("LBP", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     y=y+1
 
    # crear patches de la imagen original, 4 divisiones iguales en horizontal y vertical
    sub_img = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) 
                              for y in range(0,img.shape[1],N)]
    lbp_aux=[]
    # para cada sub imagen, calcular LBP
    for img in sub_img:
       lbp=local_binary_pattern(img,P,radius,  method='nri_uniform')
      # x = itemfreq(descriptor.ravel())
      # # Normalize the histogram
      # hist = x[:, 1]/sum(x[:, 1])
       n_bins = int(lbp.max() + 1)
       (hist, _) = np.histogram(lbp.ravel(),density=True,bins=n_bins,range=(0, n_bins))
      #    # normalize the histogram
      # hist = hist.astype("float")
      # hist /= (hist.sum() + eps)
      
      # Concatenar todos los  histogramas LBP generados
       lbp_aux.extend(hist)
    Lbp.append(lbp_aux)  
    n+=1
    
#------------------------------------
    
# #Obtener etiquetas
labels = pd.read_csv("fair_ages.csv",header=0)

# construir un DF con las caracteristicas y etiquetas
datos=pd.DataFrame(Lbp)
# necesario hacer unpack labels con **
datos= datos.assign(**labels)

#datos=datos.drop(23708,axis=0)

datos=datos.fillna(0) # Reemplazar valores NaN con ceros

# # # verificar que no hay filas que contengan valores nulos
nan_rows = datos[datos.isnull().any(1)]
print(nan_rows)

# # # # data=data.drop([6156,10840,17878,21413,22579],axis=0)

# # # # Se eliminan dichas filas
# # datos= datos.dropna(how='any')

# # # Guardar características y etiquetas en un csv
print("Guardando características...")
datos.to_csv('fair_face/LBP_96_8.csv',index=False)

# imprimir última imagen

# cv2.imshow("LBP", lbp.astype("uint8"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


