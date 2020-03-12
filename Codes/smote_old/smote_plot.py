import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load(folder_name, data_num, num_intervalos):
    X = []
    for i in range(1,data_num+1):
        #Reproduzo o nome do arquivo csv e seu caminho para ser buscado
        name_file = ''.join([folder_name,'data_adjust-',str(i),'.csv'])
        #Leio o csv
        dataset_adj = pd.read_csv(name_file)
        #Mx_thr = dataset_thr.iloc[:,10].values
        #My_thr = dataset_thr.iloc[:,11].values
        #Dados de torque retirados padronizados para 3
        #Mx = dataset_adj.iloc[:num_intervalos,10].values/3
        #My = dataset_adj.iloc[:num_intervalos,11].values/3
        Mz = dataset_adj.iloc[:num_intervalos,12].values/3
        #Fx_thr = dataset_thr.iloc[:,7].values
        #Fy_thr = dataset_thr.iloc[:,8].values
        #Dados de força retirados padronizados para 30
        #Fx = dataset_adj.iloc[:num_intervalos,7].values/30
        #Fy = dataset_adj.iloc[:num_intervalos,8].values/30
        Fz = dataset_adj.iloc[:num_intervalos,9].values/30
        #Gero um vetor em que primeiro insiro  todos os dados de Fx, depois Fy e assim por diante
        #aux = np.concatenate([Fx, Fy, Fz, Mz, My, Mx])
        aux = np.concatenate([Fz, Mz])
        #Adiciono elas no final de X formando uma matriz
        X.append(aux)
    
    #Passo X para array - possuira dimensao 2 (primeiro indice o experimento, segundo a força/torque)
    X = np.array(X)
    #Elaboro o nome do arquivo csv e seu caminho para o label.csv
    name_label = ''.join([folder_name,'labels.csv'])
    #Leio o csv
    dataset_label = pd.read_csv(name_label)
    #Pego somente a coluna de classificação dos experimentos
    y = dataset_label.iloc[:,1].values
    return X, y

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


def separate_class(X, y, data_num):
    y = y.reshape((data_num))
    X_mont = []
    X_jam = []
    X_nmont = []
    for i in range(0,data_num):
        if y[i] == 1:
            X_mont.append(X[i])
        elif y[i] == 2:
            X_jam.append(X[i])
        else :
            X_nmont.append(X[i])
    return X_mont, X_jam, X_nmont

def resample_smote(X_train, y_train, percent_mont, percent_jam, percent_notm, random_seed):
    #percent_mont, percent_jam, percent_notm = 0, 0.2, None
    X_mont_train, X_jam_train, X_nmont_train = separate_class(X_train, y_train, X_train.shape[0])
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

time = []
for i in range(122):
    time.append(np.arange(0.0, 30.710, 0.012))
    
time = np.array(time)
data_num = 500
num_int = 2560
random_seed = 8

X, y = load('dataset/',data_num,num_int)

X_mont, X_jam, X_nmont = separate_class(X,y,data_num)
X_jam = np.array(X_jam)


X_over_jam, y_over_jam = resample_smote(X, y,None,2,None,random_seed)
X_over_jam = X_over_jam[500:622,:]

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

fig, ax0 = plt.subplots(1, 1)
ax0.plot(time[0:X_jam.shape[0],:],X_jam[:,0:2560],color='blue',label='Original Sample' )
ax0.plot(time,X_over_jam[:,0:2560],color='red',label='SMOTe Sample')
#ax0.legend(loc='upper left')
ax0.axis([0,3.2,-1.2,0.25])
ax0.set_ylabel('Fz')
ax0.set_xlabel('time')
legend_without_duplicate_labels(ax0)
plt.title('Jammed Labels')
plt.show()

fig, ax1 = plt.subplots(1, 1)
ax1.plot(time[0:X_jam.shape[0],:],X_jam[:,2560:5120],color='blue',label='Original Sample')
ax1.plot(time,X_over_jam[:,2560:5120],color='red',label='SMOTe Sample')
#ax1.legend(loc='upper left')
ax1.axis([0,3.2,-0.15,0.15])
ax1.set_ylabel('Mz')
ax1.set_xlabel('time')
legend_without_duplicate_labels(ax1)
plt.title('Jammed Labels')
plt.show()

###############################################################################


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
