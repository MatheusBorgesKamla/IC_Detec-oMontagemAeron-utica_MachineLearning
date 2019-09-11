import numpy as np
import preprocessing as pre
import pandas as pd
# --- Salvando csvs remodelados para ser utilizado no WEKA ---
#Funcao que salva matriz no formato csv realizando as modificacoes para que seja aceita no WEKA
def weka_data(name_file, name_file_fin, matrix):
    #Transformando matrix em dataframe
    df_all = pd.DataFrame(data=matrix)
    #Escrevendo em arquivo csv
    df_all.to_csv(name_file)
    
    '''#Utilizo a estrutura em baixo para retirar as virgular no final de cada linha
    with open('data_write/all_exp.csv', 'r') as r, open('data_write/all_exp_not_commas.csv', 'w') as w:    
        for num, line in enumerate(r):
            if num > 0:            
                newline = line[:-2] + "\n" if "\n" in line else line[:-1]
            else:
                newline = line               
            w.write(newline)'''
    #Utilizo para retirar a primeira coluna do csv que estava bugando na hora de abrir no WEKA 
    with open(name_file, 'r') as r, open(name_file_fin, 'w') as w:    
        for num, line in enumerate(r):
            if num <= 10:
                if num != 0:         
                    newline = line[2:]
                else:
                    #newline = line
                    newline = line[1:]
            elif num <= 99:
                newline = line[3:]
            else:
                newline = line[4:]
            w.write(newline)
                
           # elif num <= 1000:
            #    newline = line[4:]
            #else:
             #   newline = line[5:]
            #w.write(newline)'''
    
    '''with open('data_write/labels.csv', 'r') as r, open('data_write/labels_fin.csv', 'w') as w:
        for num, line in enumerate(r):
            if num <= 10:
                if num != 0:         
                    newline = line[2:]
                else:
                    newline = line[1:]
            elif num <= 99:
                newline = line[3:]
            else:
                newline = line[4:]
            w.write(newline)'''
    return

data_num = 500
num_int = 2560
#Lendo todos os dados do experimento
X, y = pre.load('dataset/',data_num,num_int)
#Gerando vetor de indices
'''index = []
#Gerando lista de indices da coluna
for i in range(0,num_int):
        index.append('Fz'.join(['',str(i+1)]))
for i in range(0,num_int):
        index.append('Mz'.join(['',str(i+1)]))
index.append('class')'''
#X = X.T
fin_matrix = np.column_stack((X,y))
weka_data('data_write/all_exp.csv', 'data_write/all_exp_fin.csv', fin_matrix)
Fmax_X = np.zeros(data_num)
Fmin_X = np.zeros(data_num)
Mmax_X = np.zeros(data_num)
Mmin_X = np.zeros(data_num)
for i in range (0, data_num):
    Fmax_X[i] = max(X[i,0:num_int])
    Fmin_X[i] = min(X[i,0:num_int])
    Mmax_X[i] = max(X[i,num_int:2*num_int])
    Mmin_X[i] = min(X[i,0:num_int:2*num_int])
    
max_min = np.column_stack((Fmax_X,Fmin_X,Mmax_X,Mmin_X,y))
