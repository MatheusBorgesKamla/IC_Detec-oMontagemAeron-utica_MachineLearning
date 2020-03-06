from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def load_dataframe(folder_name, num_interval):
    #num_interval = 1555
    #folder_name = 'dataset/'
    X = []
    y = []
    names_X = ['X_train.csv', 'X_test.csv']
    names_y = ['y_train.csv', 'y_test.csv']
    for name in names_y:
        dataframe = pd.read_csv(''.join([folder_name,name]))
        aux = np.array(dataframe.iloc[:,1].values)
        y.append(aux)
        
    for name in names_X:
         dataframe = pd.read_csv(''.join([folder_name,name]))
         fz = np.array(dataframe.iloc[:,1:num_interval + 1].values)
         mz = np.array(dataframe.iloc[:,(3*num_interval) + 1: (4*num_interval) + 1].values)
         aux = np.column_stack((fz,mz))
         X.append(aux)
    
    return X[0], X[1], y[0], y[1]

def load_dataframe_allcompenents(folder_name, num_interval):
    #num_interval = 1555
    #folder_name = 'dataset/'
    X = []
    y = []
    names_X = ['X_train.csv', 'X_test.csv']
    names_y = ['y_train.csv', 'y_test.csv']
    for name in names_y:
        dataframe = pd.read_csv(''.join([folder_name,name]))
        aux = np.array(dataframe.iloc[:,1].values)
        y.append(aux)
 
    for name in names_X:
         dataframe = pd.read_csv(''.join([folder_name,name]))
         fz = np.array(dataframe.iloc[:,1:num_interval + 1].values)
         fy = np.array(dataframe.iloc[:,num_interval+1 : (2*num_interval) + 1].values)
         fx = np.array(dataframe.iloc[:,(2*num_interval) + 1: (3*num_interval) + 1].values)
         mz = np.array(dataframe.iloc[:,(3*num_interval) + 1: (4*num_interval) + 1].values)
         my = np.array(dataframe.iloc[:,(4*num_interval) + 1: (5*num_interval) + 1].values)
         mx = np.array(dataframe.iloc[:,(5*num_interval) + 1: (6*num_interval) + 1].values)
         aux = np.column_stack((fz,fy,fx,mz,my,mx))
         X.append(aux)
    
    return X[0], X[1], y[0], y[1]

def loadX_resample_allcomp(folder_name, num_interval):
    X = []
    dataframe = pd.read_csv(''.join([nameX]))
    fz = np.array(dataframe.iloc[:,0:num_interval + 1].values)
    fy = np.array(dataframe.iloc[:,num_interval+1 : (2*num_interval) + 1].values)
    fx = np.array(dataframe.iloc[:,(2*num_interval) + 1: (3*num_interval) + 1].values)
    mz = np.array(dataframe.iloc[:,(3*num_interval) + 1: (4*num_interval) + 1].values)
    my = np.array(dataframe.iloc[:,(4*num_interval) + 1: (5*num_interval) + 1].values)
    mx = np.array(dataframe.iloc[:,(5*num_interval) + 1: (6*num_interval) + 1].values)
    aux = np.column_stack((fz,fy,fx,mz,my,mx))
    X.append(aux)
    return X[0]

def loady_resample_allcomp(folder_name):
    y = []
    dataframe = pd.read_csv(''.join([namey]))
    aux = np.array(dataframe.iloc[:,0].values)
    aux = aux.astype(np.int)
    y.append(aux)
    return y[0]

def standardize_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    #Caso possuir 3 dimensoes o X_train e X_test
    if len(X_train.shape) == 3:
        for i in range(0,X_train.shape[0]):
            X_train[i] = sc.fit_transform(X_train[i])
        if X_test is not None:
            for i in range(0,X_test.shape[0]):
                X_test[i] = sc.transform(X_test[i])
    else:
        X_train = sc.fit_transform(X_train)
        if X_test is not None:
            X_test = sc.transform(X_test)
    return X_train, X_test

X_train, X_test, y_train, y_test = load_dataframe_allcompenents('dataset/', 1556)
X_train, X_test = standardize_data(X_train,X_test)

y_dummy = y_train.reshape(-1, 1)
onehotencoder = OneHotEncoder()
y_dummy = onehotencoder.fit_transform(y_dummy).toarray()

mlp_cls = Sequential()
mlp_cls.add(Dense(units=128, kernel_initializer='uniform', activation='sigmoid', input_dim=X_train.shape[1]))
mlp_cls.add(Dropout(0.5))
mlp_cls.add(Dense(units=y_dummy.shape[1], kernel_initializer='uniform', activation='softmax'))
mlp_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

weigths = mlp_cls.get_weights()

epochs = 200
mlp_cls.fit(X_train, y_dummy, epochs=epochs, batch_size=10, verbose=2)
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
y_pred_fin = y_pred_fin.astype(np.int) 

cm = confusion_matrix(y_test, y_pred_fin)
acc_test = (cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm)
acc_jam = cm[1,1]/(cm[1,0] + cm[1,1] + cm[1,2])
print('\n\nConfusion Matrix - Test: \n', cm)
print('Accuracy: ',acc_test)

acc_test_vec = []
acc_jam_vec = []

acc_test_vec.append(acc_test)
acc_jam_vec.append(acc_jam)

folder_name = 'resample_jammed_adasyn/'
percent_jam_list = np.arange(0.25,4.25,0.25)


for percent in percent_jam_list:
    nameX = ''.join([folder_name,'X_train_',str(int(percent*100)),'%_jammed.csv'])
    namey = ''.join([folder_name,'y_train_',str(int(percent*100)),'%_jammed.csv'])
    print('#################',nameX,'#################')
          
    X_aux = loadX_resample_allcomp(nameX,1556)
    y_aux = loady_resample_allcomp(namey)
    
    X_aux, X_test = standardize_data(X_aux,X_test)
    y_aux = y_aux.reshape(-1,1)
    y_aux = onehotencoder.fit_transform(y_aux).toarray()
    
    mlp_cls.set_weights(weigths)
    
    mlp_cls.fit(X_aux, y_aux, epochs=epochs, batch_size=10, verbose=2)
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
     
    cm = confusion_matrix(y_test, y_pred_fin)
    acc_test = (cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm)
    acc_jam = cm[1,1]/(cm[1,0] + cm[1,1] + cm[1,2])
    print('\n\nConfusion Matrix - Test: \n', cm)
    print('Accuracy: ',acc_test)
    acc_test_vec.append(acc_test)
    acc_jam_vec.append(acc_jam)

