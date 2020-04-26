import pandas as pd
import numpy as np

## Data path handlers
TRAIN_TEST_SET_PATH = 'dataset/'
META_DATA_PATH = '../../Data/meta/'
AUG_SET_PATH = '../../Data/aug_data_all/'
MODEL_PATH = 'models/'
IMAGES_PATH = '../../Images/'

## Selecting the desired features
parameters = ['fz','mz']

### Getting data
X_train = pd.read_csv(TRAIN_TEST_SET_PATH+'X_train.csv',index_col=0)
y_train = pd.read_csv(TRAIN_TEST_SET_PATH+'y_train.csv',index_col=0)
X_test = pd.read_csv(TRAIN_TEST_SET_PATH+'X_test.csv',index_col=0)
y_test = pd.read_csv(TRAIN_TEST_SET_PATH+'y_test.csv',index_col=0)

f_z = X_train.iloc[:, X_train.columns.str.contains(parameters[0])]
f_z = f_z/30
m_z = X_train.iloc[:, X_train.columns.str.contains(parameters[1])]
m_z = m_z/3
frames = [f_z, m_z]
X_train = pd.concat(frames, axis=1)
f_z = X_test.iloc[:, X_test.columns.str.contains(parameters[0])]
f_z = f_z/30
m_z = X_test.iloc[:, X_test.columns.str.contains(parameters[1])]
m_z = m_z/3
frames = [f_z, m_z]
X_test = pd.concat(frames, axis=1)

X_train['labels'] = y_train.copy()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=27)
for train, val in split.split(X_train, X_train['labels']):
    X_train_vl = X_train.iloc[train].copy()
    X_val = X_train.iloc[val].copy()
    
y_train_vl = X_train_vl['labels'].copy()
y_val = X_val['labels'].copy()

X_train_vl = X_train_vl.iloc[:, ~X_train_vl.columns.str.contains('labels')]
X_val = X_val.iloc[:, ~X_val.columns.str.contains('labels')]
X_train = X_train.iloc[:, ~X_train.columns.str.contains('labels')]
X_train = np.array(X_train)
X_test = np.array(X_test)

y_train_vl = np.array(y_train_vl)
y_val = np.array(y_val)
y_test = np.array(y_test)
y_train = np.array(y_train)

### Model sketch
def  build_model(C=1,gamma=0.01,kernel='rbf'):
    from sklearn.svm import SVC
    model = SVC(C = 1, kernel = 'rbf', probability=True, gamma=0.01, verbose=True)
    return model

svm_classifier = build_model()

#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

params = {
    'C': [0.1,0.5,1,5,10,50,100],
    'gamma': [0.001,0.005,0.01,0.05,0.1,0.5,1],
    'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
}

rnd_search = GridSearchCV(svm_classifier, params, cv=3, n_jobs=-1, verbose=3)

rnd_search.fit(X_train_vl, y_train_vl)

rnd_search.best_score_
rnd_search.best_params_

best_parameters = pd.DataFrame(rnd_search.best_params_, index=['values'])
best_parameters.to_csv(MODEL_PATH+'svm_original_data.csv')

### Training the reference model
best = rnd_search.best_estimator_
best.fit(X_train, y_train)
y_predict = best.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_predict)
print('\n\nConfusion Matrix: \n', cm)
acc = accuracy_score(y_test, y_predict)
print('Accuracy: ', acc)
from sklearn.externals import joblib
joblib.dump(best,MODEL_PATH+"svm_original_data.pkl")


#Error analysis
#Checking performance improvement feeding a stream of data
### Here we're going to retrain the model for each data set size.
### ranging from 1 to len(X_train)
stats = []
## The amount of data added for each training
data_batch = 2
number_of_steps = int(X_train.shape[0]/data_batch)
## Recreating the optimized model but with re-initialized weights in order to fit them with each data size
model_param = pd.read_csv(MODEL_PATH+'svm_original_data.csv',index_col=0)
C = model_param['C'].iloc[0]
gamma = model_param['gamma'].iloc[0]
kernel = model_param['kernel'].iloc[0]


for i in range(1, number_of_steps):
    svm_model = joblib.load(MODEL_PATH+"svm_original_data.pkl")
    
    if i*data_batch > X_train.shape[0]-1:
    
        svm_model.fit(X_train[:X_train.shape[0]-1,:], y_train[:X_train.shape[0]-1])
        acc = accuracy_score(y_train[:X_train.shape[0]-1],  svm_model.predict(X_train[:X_train.shape[0]-1,:]))
        acc_value = accuracy_score(y_test,   svm_model.predict(X_test))
        stats.append([acc,acc_value])
        break
    
    svm_model.fit(X_train[:i*data_batch,:], y_train[:i*data_batch])
    acc = accuracy_score(y_train[:i*data_batch],  svm_model.predict(X_train[:i*data_batch,:]))
    acc_value = accuracy_score(y_test,   svm_model.predict(X_test))
    stats.append([acc,acc_value])

stats_aux = np.array(stats) 
header = ['accuracy','val_accuracy']
stats_pd = pd.DataFrame(stats_aux,columns=header)
stats_pd.to_csv(MODEL_PATH+'stats_svm.csv')

from matplotlib import pyplot as plt

train_acc = np.array([1])
test_acc = np.array([0])

acc = np.array(stats_pd['accuracy'])
val_acc = np.array(stats_pd['val_accuracy'])
mean_step = 20
for i in range(int(acc.shape[0]/mean_step)):
    if (i+1)*mean_step > acc.shape[0]-1:
            acc_mean = acc[i*mean_step:acc.shape[0]-1].mean()
            val_acc_mean = acc[i*mean_step:val_acc.shape[0]-1].mean()
            
            train_acc = np.concatenate([train_acc, acc_mean], axis=0)
            test_acc = np.concatenate([test_acc, val_acc_mean], axis=0)
            break
    
    acc_mean = acc[i*mean_step:(i+1)*mean_step].mean()
    val_acc_mean = val_acc[i*mean_step:(i+1)*mean_step].mean()
    
        
    train_acc = np.append(train_acc, values=[acc_mean], axis=0)
    test_acc = np.append(test_acc, values=[val_acc_mean], axis=0)

### Error analysis plot
fig = plt.figure(figsize=(8,6))
fig.suptitle('Orignal Dataset Error Analysis', fontweight='bold', fontsize=14)

ax = fig.add_subplot(1,1,1)

x_axis = np.arange(train_acc.shape[0])
x_axis = mean_step*x_axis

ax.plot(x_axis, 100*train_acc, color='b', label='Train Set Accuracy')
ax.plot(x_axis, 100*test_acc, color='r', label='Test Set Accuracy')

ax.set_xlim(0, x_axis[len(x_axis)-1])
ax.set_ylim(0,100)

ax.set_xlabel('Quantity of data')
ax.set_ylabel('Accuracy %')
ax.set_title('Model performance vs quantity of data', fontweight='bold', fontsize=12)

ax.legend(loc='lower right')

plt.grid()
plt.show()
