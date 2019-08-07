import numpy as np
import preprocessing as pre
# --- SVM balanceada tomando todos os intervalos ---

data_num = 500
num_int = 2560
#Lendo todos os dados do experimento
X, y = pre.load('dataset/',data_num,num_int)

#Balanceando os dados entre os estados (aleatorizando variaveis idependentes para cada estado)
X, y = pre.proc_balanceado(X, y, data_num)
#Separando em conjunto de treino e teste (pego de forma aleatoria, aleatorizando também as variáveis dependentes)
X_train, X_test, y_train, y_test = pre.split_data(X,y,0.2,None)
#Padronizando dados
X_train, X_test = pre.standardize_data(X_train,X_test)

#Implementando a SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', probability=True, gamma='auto')
classifier.fit(X_train, y_train.ravel())

#Prevendo os resultados de teste
y_pred = classifier.predict(X_test)
svm_predict = classifier.predict_proba(X_test)

#Produzindo a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('\n\nConfusion Matrix: \n', cm)
print('\n\nAccuracy: ',(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))

#Realizando K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train.ravel(), cv = 10)
acc_mean = accuracies.mean()
acc_std = accuracies.std()
print("\n\n 10-Cross Validation: \nAccuracy Mean: ",acc_mean,"\nAccuracy Std: ",acc_std)
