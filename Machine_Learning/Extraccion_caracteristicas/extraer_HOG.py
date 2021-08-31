# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:39:53 2021

@author: Ender
"""


#----------------------------------------#
##------ Feature extraction - HOG------- ##
#----------------------------------------#

import cv2 
import os

## --------------------funciones


# variable para almacenar los nombres de las imagenes, a fin de mantenerlos
# iguales luego de alinearlas y recortarlas
files_names=[]

# Cargar imagenes desde una carpeta
def cargar_imagenes(path):
       
       images = []
       for filename in os.listdir(path):
           
         
           img = cv2.imread(os.path.join(path,filename))
           
           files_names.append(filename)
           if img is not None:
               gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
               images.append(gray)
               
       return images


##--------------------------------------
images = []
path="Imagenes/Images_resized_64x64/"
#path="UTKFace/"

Imagenes= cargar_imagenes(path)


##---------- Local Binary Pattern ----------##


from skimage.feature import  hog
import pandas as pd
radius=1 #8
P=8#16 #vecindades
Hog = []
M = Imagenes[0].shape[0]//2
N = Imagenes[0].shape[1]//2


for img in Imagenes:
 
    # crear patches de la imagen original, 4 divisiones iguales
    sub_img = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) 
                              for y in range(0,img.shape[1],N)]
    Hog_aux=[]
    # para cada sub imagen, calcular LBP
    for img in sub_img:
        hog_vector, hog_img= hog(img, orientations=10, pixels_per_cell=(8, 8),
        cells_per_block=(1,1), block_norm='L1',visualize=True,feature_vector=True,multichannel=None)
       
      # Concatenar todos los  histogramas LBP generados
        Hog_aux.extend(hog_vector)
    Hog.append(Hog_aux)  
    
# #Obtener etiquetas
labels = pd.read_csv("labels2.csv",header=0)

# construir un DF con las caracteristicas y etiquetas
datos=pd.DataFrame(Hog)
# necesario hacer unpack labels con **
datos= datos.assign(**labels)

#datos=datos.drop(23708,axis=0)

# # # verificar si hay filas que contengan valores nulos
nan_rows = datos[datos.isnull().any(1)]
print(nan_rows)

# # # # data=data.drop([6156,10840,17878,21413,22579],axis=0)

# # # # Se eliminan dichas filas
# # datos= datos.dropna(how='any')

# # # Guardar características y etiquetas en un csv
datos.to_csv('HOG_patches.csv',index=False)

# imprimir última imagen

# cv2.imshow("LBP", lbp.astype("uint8"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
