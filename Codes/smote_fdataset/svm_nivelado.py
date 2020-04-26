import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


def load_csv(path_name):
    dataframe = pd.read_csv(path_name)
    aux = dataframe.values[:,1:]
    aux = np.array(aux)
    return aux

def load_dataframe(folder_name, y_name, X_name ,num_interval, isNiv):
    X = []
    y = []

    dataframe_y = pd.read_csv(''.join([folder_name,y_name]))
    aux = np.array(dataframe_y.iloc[:,1].values)
    y.append(aux)
    
   
    dataframe_X = pd.read_csv(''.join([folder_name,X_name]))
    #Pego fz e mz somente
    fz = np.array(dataframe_X.iloc[:,1:num_interval + 1].values)/30
    if(isNiv):
        mz = np.array(dataframe_X.iloc[:,(1*num_interval) + 1: (2*num_interval) + 1].values)/3
    else:
        mz = np.array(dataframe_X.iloc[:,(3*num_interval) + 1: (4*num_interval) + 1].values)/3
    #O column_stack faz com que fz e mz se tornem uma matriz só
    aux = np.column_stack((fz,mz))
    #Adiciona em X
    X.append(aux)

    return X[0], y[0]

folder_name = 'resample_nivelado_smote/'
X_train_niv, y_train_niv = load_dataframe(folder_name, 'y_train_labels_niveladas.csv', 'X_train_labels_niveladas.csv', 1556, True)
folder_name = 'dataset/'
#X_train_niv, y_train_niv = load_dataframe(folder_name, 'y_train.csv', 'X_train.csv', 1556, False)
X_test, y_test = load_dataframe(folder_name, 'y_test.csv', 'X_test.csv', 1556, False)

##Econtrando melhores parâmetros para o modelo treinando com todas as labels
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2,n_jobs=-1)
grid.fit(X_train_niv,y_train_niv)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
cm = confusion_matrix(y_test, grid_predictions)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))
##Teste com melhores parâmetros para treinamentos com todas as labels
classifier = SVC(C = 1, kernel = 'rbf', probability=True, gamma=0.1, verbose=True)
classifier.fit(X_train_niv, y_train_niv)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))
joblib.dump(classifier,"models/svm_nivelado_allLabels.pkl")


##Separando dataset para treinar com labels 2 a 2

##Montado vs Não Montado
X_train_mn = []
y_train_mn = []
X_test_mn = []
y_test_mn = []
#Montado vs Travado
X_train_mj = []
y_train_mj = []
X_test_mj = []
y_test_mj = []
#Travado vs Não Montado
X_train_jn = []
y_train_jn = []
X_test_jn = []
y_test_jn = []

for i in range(0,y_train_niv.shape[0]):
    if(not y_train_niv[i] == 2):
        X_train_mn.append(X_train_niv[i])
        y_train_mn.append(y_train_niv[i])
    if(not y_train_niv[i] == 3):
        X_train_mj.append(X_train_niv[i])
        y_train_mj.append(y_train_niv[i])
    if(not y_train_niv[i] == 1):
        X_train_jn.append(X_train_niv[i])
        y_train_jn.append(y_train_niv[i])

for i in range(0,y_test.shape[0]):
    if(not y_test[i] == 2):
        X_test_mn.append(X_test[i])
        y_test_mn.append(y_test[i])
    if(not y_test[i] == 3):
        X_test_mj.append(X_test[i])
        y_test_mj.append(y_test[i])
    if(not y_test[i] == 1):
        X_test_jn.append(X_test[i])
        y_test_jn.append(y_test[i])


X_train_mn = np.array(X_train_mn)
y_train_mn = np.array(y_train_mn)
y_train_mn = y_train_mn/3
y_train_mn = y_train_mn.astype(int)
X_test_mn = np.array(X_test_mn)
y_test_mn = np.array(y_test_mn)
y_test_mn = y_test_mn/3
y_test_mn = y_test_mn.astype(int)
   
X_train_mj = np.array(X_train_mj)
y_train_mj = np.array(y_train_mj)
y_train_mj = y_train_mj/2
y_train_mj = y_train_mj.astype(int)
X_test_mj = np.array(X_test_mj)
y_test_mj = np.array(y_test_mj)
y_test_mj = y_test_mj/2
y_test_mj = y_test_mj.astype(int)

X_train_jn = np.array(X_train_jn)
y_train_jn = np.array(y_train_jn)
y_train_jn = y_train_jn/3
y_train_jn = y_train_jn.astype(int)
X_test_jn = np.array(X_test_jn)
y_test_jn = np.array(y_test_jn)
y_test_jn = y_test_jn/3
y_test_jn = y_test_jn.astype(int)



##Econtrando melhores parâmetros para o modelo Montado vs NaoMontado
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2,n_jobs=-1)
grid.fit(X_train_mn,y_train_mn)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test_mn)
cm = confusion_matrix(y_test_mn, grid_predictions)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1])/np.sum(cm))
##Teste com melhores parâmetros para treinamentos com Montado vs NaoMontado
classifier = SVC(C = 1, kernel = 'rbf', probability=True, gamma=0.01, verbose=True)
classifier.fit(X_train_mn, y_train_mn)
y_pred = classifier.predict(X_test_mn)
cm = confusion_matrix(y_test_mn, y_pred)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1])/np.sum(cm)) 
joblib.dump(classifier,"models/svm_nivelado_MontvsNaoMont.pkl") 

##Econtrando melhores parâmetros para o modelo Travado vs NaoMontado
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2,n_jobs=-1)
grid.fit(X_train_jn,y_train_jn)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test_jn)
cm = confusion_matrix(y_test_jn, grid_predictions)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1])/np.sum(cm))
##Teste com melhores parâmetros para treinamento Travado vs NaoMontado
classifier = SVC(C = 0.1, kernel = 'poly', probability=True, gamma=1, verbose=True)
classifier.fit(X_train_jn, y_train_jn)
y_pred = classifier.predict(X_test_jn)
cm = confusion_matrix(y_test_jn, y_pred)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1])/np.sum(cm))
joblib.dump(classifier,"models/svm_nivelado_TravadovsNaoMont.pkl")

##Econtrando melhores parâmetros para o modelo Travado vs Montado
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2,n_jobs=-1)
grid.fit(X_train_mj,y_train_mj)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test_mj)
cm = confusion_matrix(y_test_mj, grid_predictions)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1])/np.sum(cm))
##Teste com melhores parâmetros para treinamento Travado vs Montado
classifier = SVC(C = 10, kernel = 'rbf', probability=True, gamma=0.1, verbose=True)
classifier.fit(X_train_mj, y_train_mj)
y_pred = classifier.predict(X_test_mj)
cm = confusion_matrix(y_test_mj, y_pred)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1])/np.sum(cm)) 
joblib.dump(classifier,"models/svm_nivelado_TravadvsMont.pkl")


#########NIVELADO DOBRADO##################
folder_name = 'resample_nivelado_smote/'
X_train_2niv, y_train_2niv = load_dataframe(folder_name, 'y_train_labels_niveladas_dobrado.csv', 'X_train_labels_niveladas_dobrado.csv', 1556, True)

classifier = SVC(C = 1, kernel = 'rbf', probability=True, gamma=0.1, verbose=True)
classifier.fit(X_train_2niv, y_train_2niv)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))


