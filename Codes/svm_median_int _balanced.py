import numpy as np
import preprocessing as pre
# --- SVM balanceada tomando a media dos intervalos de 10 em 10 ---

data_num = 500
num_int = 2560
#Lendo todos os dados do experimento
X, y = pre.load('dataset/',data_num,num_int)
#Pegando a media em um numero de 10 intervalos para cada componente    
X = pre.med_intervalo(X,256)
#Balanceando os dados
X, y = pre.proc_balanceado(X, y, data_num)
#Separando em conjunto de treino e teste (pego de forma aleatoria, aleatorizando também as variáveis dependentes)
X_train, X_test, y_train, y_test = pre.split_data(X,y,0.2,None)
#Padronizando dados
X_train, X_test = pre.standardize_data(X_train,X_test)

#Implementando a SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', probability=True)
classifier.fit(X_train, y_train)

#Prevendo os resultados de teste
y_pred = classifier.predict(X_test)
svm_predict = classifier.predict_proba(X_test)

#Produzindo a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))




