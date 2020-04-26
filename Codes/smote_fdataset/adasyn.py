import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.utils import shuffle

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

def separate_class(X, y):
    X_mont= X[np.where(y == 1)[0]]
    X_jam = X[np.where(y == 2)[0]]
    X_nmont = X[np.where(y == 3)[0]]
    return X_mont, X_jam, X_nmont

X_train, X_test, y_train, y_test = load_dataframe('dataset/', 1556)

percent_jam_list = np.arange(0.25,4.25,0.25)

random_seed = 18

folder_save = 'resample_jammed_adasyn/'

sm = ADASYN(random_state = random_seed, n_neighbors = 4)
X_train_s, y_train_s = sm.fit_resample(X_train, y_train)

X_mont_trains, X_jam_trains, X_nmont_trains = separate_class(X_train, y_train)
jam_size = X_jam_trains.shape[0]


for percent in percent_jam_list:
    X_aux = X_train_s[0:X_train.shape[0] + int(percent*jam_size),:]
    y_aux = y_train_s[0:y_train.shape[0] + int(percent*jam_size)]
    X_aux, y_aux = shuffle(X_aux, y_aux, random_state = random_seed)
    np.savetxt(''.join([folder_save,'X_train_',str(int(percent*100)),'%_jammed.csv']),X_aux,delimiter=',')
    np.savetxt(''.join([folder_save,'y_train_',str(int(percent*100)),'%_jammed.csv']),y_aux,delimiter=',')
    
np.savetxt(''.join([folder_save,'y_test.csv']),y_test,delimiter=',')
np.savetxt(''.join([folder_save,'X_test.csv']),X_test,delimiter=',')

#########################################

X_train_all, X_test_all, y_train_all, y_test_all = load_dataframe_allcompenents('dataset/',1556)

folder_all = 'resample_jammed_adasyn/all_components/'

sm = ADASYN(random_state = random_seed, n_neighbors = 4)
X_train_s_all, y_train_s_all = sm.fit_resample(X_train_all, y_train_all)
for percent in percent_jam_list:
    X_aux = X_train_s_all[0:X_train.shape[0] + int(percent*jam_size),:]
    y_aux = y_train_s_all[0:y_train.shape[0] + int(percent*jam_size)]
    X_aux, y_aux = shuffle(X_aux, y_aux, random_state = random_seed)
    np.savetxt(''.join([folder_all,'X_train_',str(int(percent*100)),'%_jammed.csv']),X_aux,delimiter=',')
    np.savetxt(''.join([folder_all,'y_train_',str(int(percent*100)),'%_jammed.csv']),y_aux,delimiter=',')

np.savetxt(''.join([folder_all,'y_test.csv']),y_test_all,delimiter=',')
np.savetxt(''.join([folder_all,'X_test.csv']),X_test_all,delimiter=',')
