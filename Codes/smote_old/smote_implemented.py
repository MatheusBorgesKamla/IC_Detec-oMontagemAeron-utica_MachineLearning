import numpy as np
import preprocessing as pre
from math import sqrt, ceil
import random

#Calcula a distância entre dois pontos
def distance(p1, p2):
    aux = 0
    for i in range(len(p1)):
        aux += (p1[i] - p2[i])**2
        
    return sqrt(aux)

# Função que acha os n vizinhos mais próximos de um determinado elemento de um dataset
def nearest_neighbors(X, x, n):
    
    distances = []
    index = []
    
    for i in range(len(X)):
        if not np.array_equal(X[i], x):
            distances.append(distance(X[i], x))
            index.append(i)
            
    ordered_distances = sorted(distances)
    
    neighbors = []
    for i in range(n):
        neighbors.append(X[np.where(np.array(distances) == ordered_distances[i])][0])
    
    return neighbors

#Adiciona um ponto numa posição aleatória entre dois outros 
def add_random_point(p1, p2):
    
    rate = random.random()
    while rate == 0:
        rate = random.random()
        
    new_point = []
    for i in range(len(p1)):
        new_point.append(p1[i] + (p2[i] - p1[i])*rate)
        
    return np.array(new_point)
#cria amostra
def smote(X, n=1):
    datas = []
    
    for i in range(len(X)):
        nn = nearest_neighbors(X, X[i], ceil(n))
        for j in range(len(nn)):
            datas.append(add_random_point(X[i], nn[j]))
            
    while len(datas) > n*len(X):        
        del_index = np.random.randint(len(datas))
        del(datas[del_index])
            
    return np.array(datas)
#Transforma array 1D em 2D
def twod_array(array):
    return np.concatenate([array[:2560].reshape(-1,1), array[2560:].reshape(-1,1)], axis=1)
#MLP - dataset original: #######################################################################

data_num = 500
num_int = 2560
random_seed = 31

X, y = pre.load('dataset/',data_num,num_int)

y_dummy = y.reshape(-1, 1)
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y_dummy = onehotencoder.fit_transform(y_dummy).toarray()

X_train, X_test, y_train, y_test = pre.split_data(X,y_dummy,0.2,random_seed)
X_train, X_test = pre.standardize_data(X_train,X_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout


mlp_cls = Sequential()
mlp_cls.add(Dense(units=128, kernel_initializer='uniform', activation='sigmoid', input_dim=X_train.shape[1]))
mlp_cls.add(Dropout(0.5))
mlp_cls.add(Dense(units=y_train.shape[1], kernel_initializer='uniform', activation='softmax'))
mlp_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

weigths = mlp_cls.get_weights()


epochs = 200
mlp_cls.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=2)
y_predict = mlp_cls.predict(X_test)

y_pred_fin = np.zeros(y_predict.shape[0])
for i in range(0,y_predict.shape[0]):
    state = np.argmax(y_predict[i])
    if state == 0 :
        y_pred_fin[i] = 1
    elif state == 1 :
        y_pred_fin[i] = 2
    else:
        y_pred_fin[i] = 3

y_test = onehotencoder.inverse_transform(y_test) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_fin)
cm_test_nresample = cm
acc_test_nresample = (cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm)
print('\n\nConfusion Matrix - Test: \n', cm)
print('Accuracy: ',(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))

y_predict_train = mlp_cls.predict(X_train)
y_pred_train_fin = np.zeros(y_predict_train.shape[0])
for i in range(0,y_predict_train.shape[0]):
    state = np.argmax(y_predict_train[i])
    if state == 0 :
        y_pred_train_fin[i] = 1
    elif state == 1 :
        y_pred_train_fin[i] = 2
    else:
        y_pred_train_fin[i] = 3

y_train = onehotencoder.inverse_transform(y_train) 


cm2 = confusion_matrix(y_train, y_pred_train_fin)
cm_train_nresample = cm2
acc_train_nresample = (cm2[0,0]+cm2[1,1]+cm2[2,2])/np.sum(cm2)
print('\n\nConfusion Matrix - Train: \n', cm2)
print('Accuracy: ',(cm2[0,0]+cm2[1,1]+cm2[2,2])/np.sum(cm2))

file = open("smote_saves_implemented.txt","a")
text = ''.join(["-------- Original dataset: -----------\n\nConfusion Matrix - Test Samples: \n",str(cm_test_nresample),"\nAcc - Test Samples: ",str(acc_test_nresample)])
file.write(text)
text = ''.join(["\n\nConfusion Matrix - Train Samples: \n",str(cm_train_nresample),"\nAcc - Train Samples: ",str(acc_train_nresample)])
file.write(text)
#######################################################################################

percent_jam_list = np.arange(0.2,2.2,0.2)
X, y = pre.load('dataset/',data_num,num_int)

cm_test = []
acc_test = []
cm_train = []
acc_train = []

j=0

for percent in percent_jam_list:
    print("------- Jammed Resample - ",percent*100,"% ------")
    X_train, X_test, y_train, y_test = pre.split_data(X,y,0.2,random_seed)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    X_jammed = X[np.where(y == 2)[0]]
    X_jammed_smote = smote(X_jammed, percent)
    X_train = np.concatenate((X_train,X_jammed_smote))
    aux = 2*np.ones((int(percent*X_jammed.shape[0]),1),dtype=int)
    y_train = np.concatenate((y_train,aux))
    X_train, X_test = pre.standardize_data(X_train,X_test)
    y_train = onehotencoder.fit_transform(y_train).toarray()
    y_test = onehotencoder.fit_transform(y_test).toarray()
    
    mlp_cls.set_weights(weigths)
    
    mlp_cls.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=2)
    y_predict = mlp_cls.predict(X_test)
    
    y_pred_fin = np.zeros(y_predict.shape[0])
    for i in range(0,y_predict.shape[0]):
        state = np.argmax(y_predict[i])
        if state == 0 :
            y_pred_fin[i] = 1
        elif state == 1 :
            y_pred_fin[i] = 2
        else:
            y_pred_fin[i] = 3
    
    y_test = onehotencoder.inverse_transform(y_test) 
    
    
    cm = confusion_matrix(y_test, y_pred_fin)
    cm_test.append(cm)
    acc_test.append((cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))
    print('\n\nConfusion Matrix - Test: \n', cm)
    print('Accuracy: ',(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm))
    
    y_predict_train = mlp_cls.predict(X_train)
    y_pred_train_fin = np.zeros(y_predict_train.shape[0])
    for i in range(0,y_predict_train.shape[0]):
        state = np.argmax(y_predict_train[i])
        if state == 0 :
            y_pred_train_fin[i] = 1
        elif state == 1 :
            y_pred_train_fin[i] = 2
        else:
            y_pred_train_fin[i] = 3
    
    y_train = onehotencoder.inverse_transform(y_train) 
    
    
    cm2 = confusion_matrix(y_train, y_pred_train_fin)
    cm_train.append(cm2)
    acc_train.append((cm2[0,0]+cm2[1,1]+cm2[2,2])/np.sum(cm2))
    print('\n\nConfusion Matrix - Train: \n', cm2)
    print('Accuracy: ',(cm2[0,0]+cm2[1,1]+cm2[2,2])/np.sum(cm2))
    
    text = ''.join(["\n\n------ Resample Jammed - ",str(percent*100),"% ---------- \n\nConfusion Matrix - Test Samples: \n",str(cm_test[j]),"\nAcc - Test Samples: ",str(acc_test[j])])
    file.write(text)
    text = ''.join(["\n\n",str(percent*100),"% :\n\nConfusion Matrix - Train Samples: \n",str(cm_train[j]),"\nAcc - Train Samples: ",str(acc_train[j])])
    file.write(text)
    j = j + 1
file.close()
