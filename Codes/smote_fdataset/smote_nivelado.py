import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

#Função para ler dados do dataset (.csv) que recebe como parâmetros a pasta do dataset e a quantidade
#de dados de cada grandeza (fz,fy,fx ...) - para o dataset preprocessado é 1556 (a coluna 0 a 1555 é fy e assim por diante)
def load_dataframe(folder_name, num_interval):
    #num_interval = 1556
    #folder_name = 'dataset/'
    #Declaro minhas listas para armazenar valores
    X = []
    y = []
    #Lista dos nomes dos dataset preprocessado do Thiago
    names_X = ['X_train.csv', 'X_test.csv']
    names_y = ['y_train.csv', 'y_test.csv']
    #Leio primeiro y_train e depois y_test
    for name in names_y:
        #Lê o .csv e transforma em Dataframe
        dataframe = pd.read_csv(''.join([folder_name,name]))
        #Pego todos os valores (possui uma coluna só) e transforma em np array
        aux = np.array(dataframe.iloc[:,1].values)
        #Adiciono na minha lista
        y.append(aux)
    #Leio primeiro X_train e depois X_test
    for name in names_X:
        #Lê o .csv e transforma em Dataframe
         dataframe = pd.read_csv(''.join([folder_name,name]))
         #Pego fz e mz somente
         fz = np.array(dataframe.iloc[:,1:num_interval + 1].values)
         mz = np.array(dataframe.iloc[:,(3*num_interval) + 1: (4*num_interval) + 1].values)
         #O column_stack faz com que fz e mz se tornem uma matriz só
         aux = np.column_stack((fz,mz))
         #Adiciona em X
         X.append(aux)
    #Sendo assim retorno X_train, X_test, y_train, y_test
    return X[0], X[1], y[0], y[1]

#Função igual a anterior só que em vez de pegar somente fz e mz pega todas as componentes
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

#Leio os .csv
X_train, X_test, y_train, y_test = load_dataframe('dataset/', 1556)
#X_train, X_test, y_train, y_test = load_dataframe_allcompenents('dataset/', 1556)

#Separo X_train entre montado, travado e não montado
X_mont= X_train[np.where(y_train == 1)[0]]
X_jam = X_train[np.where(y_train == 2)[0]]
X_nmont = X_train[np.where(y_train == 3)[0]]

#Pasta para salvar os .csv com aumento dos dados
folder_save = 'resample_nivelado_smote/'
#folder_save = 'resample_nivelado_smote/todas_componentes/'

#Seed utilizada para a seleção randômica dos algoritmos
random_seed = 18

#Primeiramente vou fazer o aumento das outras labels sem ser a majoritária (no caso montado) a fim de ficar nivelado
sm = SMOTE(sampling_strategy='not majority', random_state=random_seed, k_neighbors=5)
#Guardo o novo dataset com as labels niveladas
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
#Como os dados sintéticos gerado são adicionados em sequência no final da matrix, realizo um embaralhamento a fim de evitar isso
X_aux, y_aux = shuffle(X_train_res, y_train_res, random_state = random_seed)

#Crio um header para salvar os .csv
index = np.arange(0,1556, 1)
parameters = ['fz','mz']
#parameters = ['fz','fy','fx','mz','my','mx']
header =  []
for parm in parameters:
    for ind in index:
        header.append(parm+str(ind))
        
#Passo X_train e y_train para dataframe e depois salvo como .csv utilizando o header criado  
X_train_df = pd.DataFrame(X_aux,columns=header)
X_train_df.to_csv(folder_save + 'X_train_labels_niveladas.csv')
y_header = ['label']
y_train_df = pd.DataFrame(y_aux,columns=y_header)
y_train_df.to_csv(folder_save + 'y_train_labels_niveladas.csv')

################################DUPLICANDO################################################
#Dicionário criado para passar de parâmetro para a função SMOTE
#Tem que passar a label e o quanto de dados você quer para tal label depois do aumento
#Primeiro vou aumentar 2 vezes só o montado para depois nivelar todo mundo
dic_resample = {1 : X_mont.shape[0]*2, 2 : X_jam.shape[0], 3: X_nmont.shape[0]}

sm = SMOTE(sampling_strategy= dic_resample, random_state=random_seed, k_neighbors=5)
#Aplico o SMOTE sobre os dados ORIGINAIS realizando o aumento de 2 vezes do montado
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
#Após dobrar o montado irei nivelar todo mundo
sm = SMOTE(sampling_strategy='not majority', random_state=random_seed, k_neighbors=5)
#Todo mundo nivelado perante o montado
X_train_resn, y_train_resn = sm.fit_resample(X_train_res, y_train_res)
#Faço um embaralhamento para evitar que todos os dados sintéticos fiquem concentrados no final da matriz
X_aux, y_aux = shuffle(X_train_resn, y_train_resn, random_state = random_seed)

#Passo para dataframe e depois salvo como .csv udando o header já criado
X_train_df = pd.DataFrame(X_aux,columns=header)
X_train_df.to_csv(folder_save + 'X_train_labels_niveladas_dobrado.csv')
y_train_df = pd.DataFrame(y_aux,columns=y_header)
y_train_df.to_csv(folder_save + 'y_train_labels_niveladas_dobrado.csv')
################################QUADRIPLICANDO###############################################
#Mesmo esquema do que já foi feito. Não criei uma função para que pudesse ir debugando os resultados linha à linha
#Mas se for aumentar mais talvez seria melhor criar uma funçãozinha
dic_resample = {1 : X_mont.shape[0]*4, 2 : X_jam.shape[0], 3: X_nmont.shape[0]}

sm = SMOTE(sampling_strategy=dic_resample, random_state=random_seed, k_neighbors=5)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

sm = SMOTE(sampling_strategy='not majority', random_state=random_seed, k_neighbors=5)

X_train_resn, y_train_resn = sm.fit_resample(X_train_res, y_train_res)

X_aux, y_aux = shuffle(X_train_resn, y_train_resn, random_state = random_seed)


X_train_df = pd.DataFrame(X_aux,columns=header)
X_train_df.to_csv(folder_save + 'X_train_labels_niveladas_quadriplicado.csv')
y_train_df = pd.DataFrame(y_aux,columns=y_header)
y_train_df.to_csv(folder_save + 'y_train_labels_niveladas_quadriplicado.csv')

pd.read_csv(name_arq, index_col=0)