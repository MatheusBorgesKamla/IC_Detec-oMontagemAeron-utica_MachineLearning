import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

num_interval = 1555
path_x = 'dataset/X_train.csv'
path_y = 'dataset/y_train.csv'
y = []
#X = []
dataframe = pd.read_csv(path_x)
fz = np.array(dataframe.iloc[:,1:num_interval + 1].values)
#fy = np.array(dataframe.iloc[:,num_interval+1 : (2*num_interval) + 1].values)
#fx = np.array(dataframe.iloc[:,(2*num_interval) + 1: (3*num_interval) + 1].values)
mz = np.array(dataframe.iloc[:,(3*num_interval) + 1: (4*num_interval) + 1].values)
#my = np.array(dataframe.iloc[:,(4*num_interval) + 1: (5*num_interval) + 1].values)
#mx = np.array(dataframe.iloc[:,(5*num_interval) + 1: (6*num_interval) + 1].values)
#X.append([fz,mz])
#X = np.array(X)
#X = X.reshape(X.shape[1],X.shape[2],X.shape[3])
dataframe = pd.read_csv(path_y)
aux = np.array(dataframe.iloc[:,1].values)
y.append(aux)
y = np.array(y)

fz_mont= fz[np.where(y == 1)[1]]
fz_jam = fz[np.where(y == 2)[1]]
fz_nmont = fz[np.where(y == 3)[1]]

mz_mont= mz[np.where(y == 1)[1]]
mz_jam = mz[np.where(y == 2)[1]]
mz_nmont = mz[np.where(y == 3)[1]]

time_step = 0.012
time = np.arange(0.0, num_interval*time_step, time_step)

fz_mont_med = np.mean(fz_mont,0)
fz_mont_std = np.std(fz_mont, 0)
fz_jam_med = np.mean(fz_jam,0)
fz_jam_std = np.std(fz_jam, 0)
fz_nmont_med = np.mean(fz_nmont,0)
fz_nmont_std = np.std(fz_nmont,0)

mz_mont_med = np.mean(mz_mont,0)
mz_mont_std = np.std(mz_mont, 0)
mz_jam_med = np.mean(mz_jam,0)
mz_jam_std = np.std(mz_jam, 0)
mz_nmont_med = np.mean(mz_nmont,0)
mz_nmont_std = np.std(mz_nmont,0)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

fig, ax0 = plt.subplots(1, 1, figsize=(9,3))

ax0.plot(time,fz_nmont_med,color='red',label='Not Mounted')
ax0.fill_between(time, fz_nmont_med - fz_nmont_std, fz_nmont_med + fz_nmont_std, color=(1,0,0,0.3))
ax0.plot(time,fz_jam_med,color='blue',label='Jammed' )
ax0.fill_between(time, fz_jam_med - fz_jam_std, fz_jam_med + fz_jam_std, color=(0,0,1,0.3))
ax0.plot(time,fz_mont_med,color=(78/255,189/255,70/255),label='Mounted')
ax0.fill_between(time, fz_mont_med - fz_mont_std, fz_mont_med + fz_mont_std, color=(78/255,189/255,70/255,0.3))
#ax0.legend(loc='upper left')
ax0.axis([-0.1,17.5,-36,0])
ax0.set_ylabel('Force (N)')
ax0.set_xlabel('Time (s)')
legend_without_duplicate_labels(ax0)

plt.show()

fig, ax1 = plt.subplots(1, 1, figsize=(9,4))

ax1.plot(time,mz_nmont_med,color='red',label='Not Mounted')
ax1.fill_between(time, mz_nmont_med - mz_nmont_std, mz_nmont_med + mz_nmont_std, color=(1,0,0,0.3))
ax1.plot(time,mz_jam_med,color='blue',label='Jammed' )
ax1.fill_between(time, mz_jam_med - mz_jam_std, mz_jam_med + mz_jam_std, color=(0,0,1,0.3))
ax1.plot(time,mz_mont_med,color=(78/255,189/255,70/255),label='Mounted')
ax1.fill_between(time, mz_mont_med - mz_mont_std, mz_mont_med + mz_mont_std, color=(78/255,189/255,70/255,0.3))
#ax0.legend(loc='upper left')
ax1.axis([-0.1,17.5,-0.2,0.4])
ax1.set_ylabel('Torque (N)')
ax1.set_xlabel('Time (s)')
legend_without_duplicate_labels(ax1)

plt.show()

path_resample_x = 'resample_jammed_smote/all_components/X_train_400%_jammed.csv'
path_resample_y = 'resample_jammed_smote/y_train_400%_jammed.csv'
dataframe = pd.read_csv(path_resample_x)
fz_resample = np.array(dataframe.iloc[:,1:num_interval + 1].values)
mz_resample = np.array(dataframe.iloc[:,(3*num_interval) + 1: (4*num_interval) + 1].values)
dataframe = pd.read_csv(path_resample_y)
aux = np.array(dataframe.iloc[:,0].values)
y_resample = []
y_resample.append(aux)
y_resample = np.array(y_resample)


fz_jam_r = fz_resample[np.where(y_resample == 2)[1]]

mz_jam_r = mz_resample[np.where(y_resample == 2)[1]]


fz_jam_r_med = np.mean(fz_jam_r,0)
fz_jam_r_std = np.std(fz_jam_r, 0)


mz_jam_r_med = np.mean(mz_jam_r,0)
mz_jam_r_std = np.std(mz_jam_r, 0)

fig, ax0 = plt.subplots(1, 1, figsize=(9,3))


ax0.plot(time,fz_jam_med,color='blue',label='Original Jammed' )
ax0.fill_between(time, fz_jam_med - fz_jam_std, fz_jam_med + fz_jam_std, color=(0,0,1,0.3))
ax0.plot(time,fz_jam_r_med,color='red',label='Oversampled Jammmed')
ax0.fill_between(time, fz_jam_r_med - fz_jam_r_std, fz_jam_r_med + fz_jam_r_std, color=(1,0,0,0.3))
#ax0.legend(loc='upper left
ax0.axis([-0.1,17.5,-38,0])
ax0.set_ylabel('Force (N)')
ax0.set_xlabel('Time (s)')
legend_without_duplicate_labels(ax0)

plt.show()

fig, ax1 = plt.subplots(1, 1, figsize=(9,4))


ax1.plot(time,mz_jam_med,color='blue',label='Original Jammed' )
ax1.fill_between(time, mz_jam_med - mz_jam_std, mz_jam_med + mz_jam_std, color=(0,0,1,0.3))
ax1.plot(time,mz_jam_r_med,color='red',label='Oversampled Jammed')
ax1.fill_between(time, mz_jam_r_med - mz_jam_r_std, mz_jam_r_med + mz_jam_r_std, color=(1,0,0,0.3))
#ax0.legend(loc='upper left')
ax1.axis([-0.1,17.5,-0.2,0.4])
ax1.set_ylabel('Torque (N)')
ax1.set_xlabel('Time (s)')
legend_without_duplicate_labels(ax1)

plt.show()

