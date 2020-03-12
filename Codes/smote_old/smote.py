import numpy as np
import preprocessing as pre



def resample_smote(X_train, y_train, percent_mont, percent_jam, percent_notm, random_seed):
    #percent_mont, percent_jam, percent_notm = 0, 0.2, None
    X_mont_train, X_jam_train, X_nmont_train = pre.separate_class(X_train, y_train, X_train.shape[0])
    sample_size_train = [len(X_mont_train), len(X_jam_train), len(X_nmont_train)]


    percent_all = [percent_mont, percent_jam, percent_notm]
    for i in range(0,3):
        if percent_all[i] is None:
            percent_all[i] = 0

    K=0
    for i in percent_all:
        if i > 1:
            K = 2
        else:
            K = 1
            
    resample_size = []
    for i in range(0,3):
        resample_size.append(int(percent_all[i]*sample_size_train[i]) + sample_size_train[i])
    
    dic_resample_size = {1 : resample_size[0], 2 : resample_size[1], 3 : resample_size[2] }
    
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(sampling_strategy=dic_resample_size, random_state = random_seed, k_neighbors = K)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        
    return X_train_res, y_train_res

#MLP - dataset original: #######################################################################

data_num = 500
num_int = 2560
random_seed = 8

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

file = open("smote_saves.txt","a")
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
    X_train, y_train = resample_smote(X_train, y_train,None,percent,None,random_seed)
    X_train, X_test = pre.standardize_data(X_train,X_test)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
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
