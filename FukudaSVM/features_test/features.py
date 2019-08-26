import numpy as np
import pandas as pd


def load_thr(folder_name, data_num):
    Mz_tot_thr = []
    Fz_tot_thr = []
    Fz_max = []
    for i in range(1,data_num+1):
        name_file = ''.join([folder_name,'data_threading-',str(i),'.csv'])
        dataset_thr = pd.read_csv(name_file)
        #Mx_thr = dataset_thr.iloc[:,10].values
        #My_thr = dataset_thr.iloc[:,11].values
        Mz_thr = dataset_thr.iloc[:,12].values
        Mz_tot_thr = np.concatenate([Mz_tot_thr, Mz_thr])        
        #Fx_thr = dataset_thr.iloc[:,7].values
        #Fy_thr = dataset_thr.iloc[:,8].values
        Fz_thr = dataset_thr.iloc[:,9].values
        Fz_max.append(max(Fz_thr))
        Fz_tot_thr = np.concatenate([Fz_tot_thr, Fz_thr])

    Fz_max = np.array(Fz_max)
    return Fz_tot_thr, Mz_tot_thr, Fz_max


data_num = 480
Fz, Mz, Fz_max = load_thr('data/',data_num)




        