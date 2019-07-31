import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

'''dataset_adj = pd.read_csv('data/data_adjust-1.csv')
Mx = np.concatenate( (dataset_adj.iloc[:,10].values,dataset_thr.iloc[:,10].values))
My = np.concatenate( (dataset_adj.iloc[:,11].values,dataset_thr.iloc[:,11].values))
Mz = np.concatenate( (dataset_adj.iloc[:,12].values,dataset_thr.iloc[:,12].values))
time = np.concatenate( (dataset_adj.iloc[:,0].values,dataset_thr.iloc[:,0].values))'''

'''Mx_adj = dataset_adj.iloc[:,10].values
My_adj = dataset_adj.iloc[:,11].values
Mz_adj = dataset_adj.iloc[:,12].values
time_adj = dataset_adj.iloc[:,0].values

M_adj = Mx_adj

for i in range(0,time_adj.shape[0]):
    M_adj[i] = ((Mx_adj[i] ** 2) + (My_adj[i] ** 2) + (Mz_adj[i] ** 2)) ** 1/2  

plt.plot(time_adj,M_adj)
plt.title('Torque - Adjustment Action')
plt.xlabel('Time')
plt.ylabel('Torque')
plt.show()


dataset_thr = pd.read_csv('data/data_threading-1.csv')
Mx_thr = dataset_thr.iloc[:,10].values
My_thr = dataset_thr.iloc[:,11].values
Mz_thr = dataset_thr.iloc[:,12].values
time_thr = dataset_thr.iloc[:,0].values


M_thr = Mx_thr

for i in range(0,time_thr.shape[0]):
    M_thr[i] = ((Mx_thr[i] ** 2) + (My_thr[i] ** 2) + (Mz_thr[i] ** 2)) ** 1/2  


plt.plot(time_thr,M_thr)

plt.title('Torque - Threading Action')
plt.xlabel('Time')
plt.ylabel('Torque')
'''


data_num = 360

plt.figure(1)
for i in range(1,data_num+1):
    name_file = ''.join(['data/data_threading-',str(i),'.csv'])
    dataset_thr = pd.read_csv(name_file)
    #Mx_thr = dataset_thr.iloc[:,10].values
    #My_thr = dataset_thr.iloc[:,11].values
    Mz_thr = dataset_thr.iloc[:,12].values
    time_thr = dataset_thr.iloc[:,0].values
    #M_thr = Mx_thr
    '''for j in range(0,time_thr.shape[0]):
        M_thr[j] = ((Mx_thr[j] ** 2) + (My_thr[j] ** 2) + (Mz_thr[j] ** 2)) ** 1/2'''
    plt.plot(time_thr,Mz_thr)

plt.title('Torque - Threading Action')
plt.xlabel('Time')
plt.ylabel('Torque')
plt.show()

plt.figure(2)
for i in range(1,data_num+1):
    name_file = ''.join(['data/data_threading-',str(i),'.csv'])
    dataset_thr = pd.read_csv(name_file)
    Mz_thr = dataset_thr.iloc[:,12].values
    Mz_thr_min = min(Mz_thr)
    Mz_thr_max = max(Mz_thr)
    plt.scatter(abs(Mz_thr_min),Mz_thr_max)

plt.title('Torque - Threading Action')
plt.xlabel('Torque |min|')
plt.ylabel('Torque max')
#plt.xlim(0.18,0.6)
#plt.ylim(0.1,0.75)
plt.show()
    
plt.figure(3)
for i in range(1,data_num+1):
    name_file = ''.join(['data/data_threading-',str(i),'.csv'])
    dataset_thr = pd.read_csv(name_file)
    Mz_thr = dataset_thr.iloc[:,12].values
    Mz_thr_min = min(Mz_thr)
    Mz_thr_max = max(Mz_thr)
    Mz_thr_dif = (Mz_thr_max - Mz_thr_min)
    plt.scatter(Mz_thr_dif,Mz_thr_max)

plt.title('Torque - Threading Action')
plt.xlabel('Torque (max-min)')
plt.ylabel('Torque max')
#plt.xlim(0.18,1.25)
#plt.ylim(0.1,0.8)
plt.show()

data_num_adj = 100
dataset_label = pd.read_csv('data/labels.csv')


labels = dataset_label.iloc[:,1].values
fig = plt.figure(4)
ax  = fig.add_subplot(111)
for i in range(1,data_num_adj+1):
    name_file = ''.join(['data/data_adjust-',str(i),'.csv'])
    dataset_adj = pd.read_csv(name_file)
    #Mx_adj = dataset_adj.iloc[:,10].values
    #My_adj = dataset_adj.iloc[:,11].values
    Mz_adj = dataset_adj.iloc[:,12].values
    time_adj = dataset_adj.iloc[:,0].values
    #M_adj = Mx_adj
    '''for j in range(0,time_thr.shape[0]):
        M_thr[j] = ((Mx_thr[j] ** 2) + (My_thr[j] ** 2) + (Mz_thr[j] ** 2)) ** 1/2'''
    if labels[i] == 1 :
         plt.plot(time_adj,Mz_adj,color='green',label='1')
    elif labels[i] == 2 :
        plt.plot(time_adj,Mz_adj,color='blue',label='2')
    elif labels[i] == 3 :
        plt.plot(time_adj,Mz_adj,color='red',label='3')
    elif labels[i] == 4 :
        plt.plot(time_adj,Mz_adj,color='yellow',label='4')

plt.title('Torque - Adjusting Action')
plt.xlabel('Time')
plt.ylabel('Torque')
#Para nao repetir a legenda
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
    if l not in lablout:
        lablout.append(l)
        handout.append(h)
fig.legend(handout, lablout)
plt.show()

fig = plt.figure(5)
for i in range(1,data_num_adj+1):
    name_file = ''.join(['data/data_adjust-',str(i),'.csv'])
    dataset_adj = pd.read_csv(name_file)
    Mz_adj = dataset_adj.iloc[:,12].values
    Mz_adj_min = min(Mz_adj)
    Mz_adj_max = max(Mz_adj)
    if labels[i] == 1 :
         plt.scatter(abs(Mz_adj_min),Mz_adj_max,color='green',label='1')
    elif labels[i] == 2 :
        plt.scatter(abs(Mz_adj_min),Mz_adj_max,color='blue',label='2')
    elif labels[i] == 3 :
        plt.scatter(abs(Mz_adj_min),Mz_adj_max,color='red',label='3')
    elif labels[i] == 4 :
        plt.scatter(abs(Mz_adj_min),Mz_adj_max,color='yellow',label='4')

plt.title('Torque - Adjusting Action')
plt.xlabel('Torque |min|')
plt.ylabel('Torque max')
#plt.xlim(0.18,0.6)
#plt.ylim(0.1,0.75)
fig.legend(handout, lablout)
plt.show()

fig = plt.figure(6)
for i in range(1,data_num_adj+1):
    name_file = ''.join(['data/data_adjust-',str(i),'.csv'])
    dataset_adj = pd.read_csv(name_file)
    Mz_adj = dataset_adj.iloc[:,12].values
    Mz_adj_min = min(Mz_adj)
    Mz_adj_max = max(Mz_adj)
    Mz_adj_dif = Mz_adj_max - Mz_adj_min
    if labels[i] == 1 :
         plt.scatter(Mz_adj_dif,abs(Mz_adj_min),color='green',label='1')
    elif labels[i] == 2 :
        plt.scatter(Mz_adj_dif,abs(Mz_adj_min),color='blue',label='2')
    elif labels[i] == 3 :
        plt.scatter(Mz_adj_dif,abs(Mz_adj_min),color='red',label='3')
    elif labels[i] == 4 :
        plt.scatter(Mz_adj_dif,abs(Mz_adj_min),color='yellow',label='4')

plt.title('Torque - Adjusting Action')
plt.xlabel('Torque (max-min)')
plt.ylabel('|Torque min|')
#plt.xlim(0.18,0.6)
#plt.ylim(0.1,0.75)
fig.legend(handout, lablout)
plt.show()
    

    
    
     
 
