import numpy as np
import preprocessing as pre
# --- LSTM balanceada tomando todos os intervalos ---

data_num = 500
num_int = 2560
#Lendo todos os dados do experimento
X, y = pre.load_3dim('dataset/',data_num,num_int)
#Balanceando os dados entre os estados (aleatorizando variaveis idependentes para cada estado)
X, y = pre.proc_balanceado(X, y, data_num)
#Remodelado as dimensões de y para ser aceito na dummy
y = np.reshape(y,(y.shape[0],-1))
#Passando y para dummy variables
y_dummy = pre.dummy_variables(y)
#Separando em conjunto de treino e teste (pego de forma aleatoria, aleatorizando também as variáveis dependentes)
X_train, X_test, y_train, y_test = pre.split_data(X,y_dummy,0.2,None)
#Padronizando dados
X_train, X_test = pre.standardize_data(X_train,X_test)
#Implementando a LSTM
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
#Dimensao da camada invisivel
hidden_size = 32
#Criando obbjeto da rede
sl_model = Sequential()
#gerando uma camada do tipo LSTM que recebe o numero de saidas, o tipo de funcao de ativacao geralmente a tangente hiperbolica
# o quanto irei desconsiderar dos dados de entrada nessa camada e quanto irei desconsideradar do estado de recorrencia anterior
sl_model.add(LSTM(units=hidden_size, input_shape=(X_train.shape[1],X_train.shape[2]) ,activation='tanh', dropout=0.2, recurrent_dropout=0.2))
#Adicionando a camanda de saida como uma camada comum densely-connected NN que possuira uma saida e uma funcao de ativacao sigmoid
sl_model.add(Dense(units=y_train.shape[1], activation='sigmoid'))
#.compile ira realizar a configuracao final da rede para que possa sser treinada
sl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 5


sl_model.fit(X_train, y_train, epochs=epochs, shuffle=True)
y_predict = sl_model.predict(X_test)

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

#Realizando K-fold Cross Validation com 10 folders
from sklearn.model_selection import StratifiedKFold
cvscores =[]
kfold = StratifiedKFold(n_splits=10, shuffle=True)
for train, test in kfold.split(X[:,0,0], y_dummy[:,0]):
    sl_model = Sequential()
    sl_model.add(LSTM(units=hidden_size, input_shape=(X_train.shape[1],X_train.shape[2]) ,activation='tanh', dropout=0.2, recurrent_dropout=0.2))
    sl_model.add(Dense(units=y_train.shape[1], activation='sigmoid'))
    sl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    sl_model.fit(X[train], y_dummy[train], epochs=1, batch_size=10, verbose=0, validation_data=(X[test], y_dummy[test]))
    scores = sl_model.evaluate(X[test], y_dummy[test], verbose=0)
    print("%s: %.2f%%" % (sl_model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
   

