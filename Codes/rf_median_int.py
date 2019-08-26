import numpy as np
import preprocessing as pre
# --- Random Forest nao balanceada tomando todos os intervalos ---

data_num = 500
num_int = 2560
#Lendo todos os dados do experimento
X, y = pre.load('dataset/',data_num,num_int)
#Pegando a media em um numero de 10 intervalos para cada componente    
X = pre.med_intervalo(X,10)
#Remodelado as dimensões de y para ser aceito na dummy
y = np.reshape(y,(y.shape[0],-1))
#Passando y para dummy variables
y_dummy = pre.dummy_variables(y)
#Separando em conjunto de treino e teste (pego de forma aleatoria, aleatorizando também as variáveis dependentes)
X_train, X_test, y_train, y_test = pre.split_data(X,y_dummy,0.2,None)
#Padronizando dados
X_train, X_test = pre.standardize_data(X_train,X_test)
#Implementando a RandomForest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = None)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

y_pred_fin = np.zeros(y_predict.shape[0])
for i in range(0,y_predict.shape[0]):
    state = np.argmax(y_predict[i])
    if state == 0 :
        y_pred_fin[i] = 1
    elif state == 1 :
        y_pred_fin[i] = 2
    else:
        y_pred_fin[i] = 3
        
#Retornando o y_test para rotulos normais para conseguir chamar a confusion matrix
y_test_fin = np.zeros(y_test.shape[0])
for i in range(0,y_test.shape[0]):
    state = np.argmax(y_test[i])
    if state == 0 :
        y_test_fin[i] = 1
    elif state == 1 :
        y_test_fin[i] = 2
    else:
        y_test_fin[i] = 3
# Produzindo a confusing matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_fin, y_pred_fin)
print('\n\nConfusion Matrix: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))

from sklearn.model_selection import cross_val_score
#Aleatorizando dados balanceados
X, X_none, y, y_none = pre.split_data(X, y, 0, None)
#Balanceando os dados entre os estados (aleatorizando variaveis idependentes para cada estado)
y = np.reshape(y,(y.shape[0],-1))
#Passando y para dummy variables
y_dummy = pre.dummy_variables(y)
accuracies = cross_val_score(estimator = classifier, X = X, y = y.ravel(), cv = 10)
acc_mean = accuracies.mean()
acc_std = accuracies.std()
print("\n\n 10-Cross Validation: \nAccuracy Mean: ",acc_mean,"\nAccuracy Std: ",acc_std)
