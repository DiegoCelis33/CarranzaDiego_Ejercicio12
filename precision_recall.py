#!/usr/bin/env python
# coding: utf-8

# In[48]:


import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve

# lee los numeros
numeros = skdata.load_digits()

# lee los labels
target = numeros['target']

# lee las imagenes
imagenes = numeros['images']

# cuenta el numero de imagenes total
n_imagenes = len(target)

# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

# Split en train/test
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

# todo lo que es diferente de 1 queda marcado como 0
y_train[y_train!=1]=0
y_test[y_test!=1]=0


# Reescalado de los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Encuentro los autovalores y autovectores de las imagenes marcadas como 1.
numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

# encuentro las imagenes en el espacio de los autovectores
x_test_transform = x_test @ vectores
x_train_transform = x_train @ vectores

linear = LinearDiscriminantAnalysis()

# numero de componentes a utilizar, de 3 a 40.
n_comp = np.arange(3,41)

# arrays para guardar valores de f1
f1_test_unos = np.ones(len(n_comp))
f1_train_unos = np.ones(len(n_comp))
f1_test_otros = np.ones(len(n_comp))
f1_train_otros = np.ones(len(n_comp))


#for i, n_components in enumerate(n_comp):
    # encuentro los parametros del clasificador
linear.fit(x_train_transform[:,:10], y_train)
    
    # predigo los valores para train
y_predict_train = linear.predict(x_train_transform[:,:10])
   
    # predigo los valores para test
y_predict_test = linear.predict(x_test_transform[:,:10])

    #calculo F1-score
    #f1_test_unos[i] = sklearn.metrics.f1_score(y_test, y_predict_test, pos_label=1)
    #f1_train_unos[i] = sklearn.metrics.f1_score(y_train, y_predict_train, pos_label=1)
    #f1_test_otros[i] = sklearn.metrics.f1_score(y_test, y_predict_test, pos_label=0)
    #f1_train_otros[i] = sklearn.metrics.f1_score(y_train, y_predict_train, pos_label=0)

    
proba_y = linear.predict_proba(x_test_transform[:,:10])[:,1]

precision, recall, thresholds = precision_recall_curve(y_test,proba_y)

f1 = 2*(precision*recall)/(precision+recall)




plt.figure()
plt.subplot(1,2,1)
plt.plot(recall,precision)
plt.xlabel("recall")
plt.ylabel("precision")
plt.subplot(1,2,2)
plt.plot(thresholds,f1[1:])
plt.xlabel("Thresholds")
plt.ylabel("F1")
plt.tight_layout()

plt.savefig("F1_prec_recall.png")


# In[31]:


np.shape(precision)


# In[ ]:




