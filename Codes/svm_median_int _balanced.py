import numpy as np
import preprocessing as pre
# --- SVM balanceada tomando a media dos intervalos de 10 em 10 ---

data_num = 500
num_int = 2560
#Lendo todos os dados do experimento
X, y = pre.load('dataset/',data_num,num_int)
#Pegando a media em um numero de 10 intervalos para cada componente    
X = pre.med_intervalo(X,10)
#Balanceando os dados
X, y = pre.proc_balanceado(X, y, data_num)
#Separando em conjunto de treino e teste (pego de forma aleatoria, aleatorizando também as variáveis dependentes)
X_train, X_test, y_train, y_test = pre.split_data(X,y,0.2,None)
#Padronizando dados
X_train, X_test = pre.standardize_data(X_train,X_test)

#Implementando a SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', probability=True, gamma='auto')
#Treinando a SVM
classifier.fit(X_train, y_train.ravel())

#Prevendo os resultados de teste 
y_pred = classifier.predict(X_test)
svm_predict = classifier.predict_proba(X_test)

#Produzindo a confusion matrix da SVM acima
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))


#Realizando K-fold Cross Validation com 10 folders
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test.ravel(), cv = 10)
acc_mean = accuracies.mean()
acc_std = accuracies.std()
print("\n\n 10-Cross Validation: \nAccuracy Mean: ",acc_mean,"\nAccuracy Std: ",acc_std)

#Realizando K-fold Cross Validation gerando confusion matrix para cada iteração

''' from sklearn.model_selection import StratifiedKFold
X_fold, X_none, y_fold, y_none = pre.split_data(X,y,0,None)
y_fold = y_fold.ravel()
X_fold, X_none = pre.standardize_data(X,None)
X_fold = X
kfold = StratifiedKFold(n_splits=10,shuffle=True)
cont = 0
acc_fold = np.zeros(10)
for train, test in kfold.split(X_fold,y_fold):
    print("\nTrain: ",train, "\nTest: ",test)
    clf = SVC(kernel= 'rbf',gamma='auto')
    clf.fit(X_fold[train],y_fold[train],)
    y_fold_pred = clf.predict(X_fold[test])
    cm = confusion_matrix(y_fold[test],y_fold_pred)
    acc_fold[cont] = (cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm)
    print("------ Folder ",cont+1, "-------- \n\nConfusion Matrix: \n",cm,"\nAccuracy: ",acc_fold[cont])
    cont += 1

print("\n\n 10-Cross Validation: \nAccuracy Mean: ",acc_fold.mean(),"\nAccuracy Std: ",acc_fold.std())'''


