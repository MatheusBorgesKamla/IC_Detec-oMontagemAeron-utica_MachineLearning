import numpy as np
import preprocessing as pre
import matplotlib.pyplot as plt
import pandas as pd

#--- Plotando graficos para visualizacao do experimento

dataset = pd.read_csv('dataset/data_adjust-1.csv')
num_int = 2560
Mz = dataset.iloc[:num_int,12].values
Fz = dataset.iloc[:num_int,9].values
time = dataset.iloc[:num_int,0].values

n_inter = 10
inter_length = int(Mz.shape[0]/(1*n_inter))
Mz_med = np.zeros(n_inter*1)
Mz_med_plt = np.zeros_like(Mz)
for i in range(0,n_inter*1):
    ind = i*inter_length
    Mz_med[i] = np.mean(Mz[ind:ind+inter_length])
    Mz_med_plt[ind:ind+inter_length] = Mz_med[i]

figM = plt.figure(1)
plt.plot(time,Mz,label='All values',color='red')
plt.plot(time,Mz_med_plt,label='Med. intervals',color='blue')
plt.title('Adjusting Action')
plt.xlabel('Time')
plt.ylabel('Torque Z-axis (Mz)')
figM.legend()
plt.show()

figF = plt.figure(2)
n_inter = 10
inter_length = int(Mz.shape[0]/(1*n_inter))
Fz_med = np.zeros(n_inter*1)
Fz_med_plt = np.zeros_like(Mz)
for i in range(0,n_inter*1):
    ind = i*inter_length
    Fz_med[i] = np.mean(Fz[ind:ind+inter_length])
    Fz_med_plt[ind:ind+inter_length] = Fz_med[i]
plt.plot(time,Fz,label='All values',color='red')
plt.plot(time,Fz_med_plt,label='Med. intervals',color='blue')
plt.title('Adjusting Action')
plt.xlabel('Time')
plt.ylabel('Force Z-axis (Mz)')
figF.legend()
plt.show()