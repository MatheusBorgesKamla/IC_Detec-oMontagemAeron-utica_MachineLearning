import pandas as pd
import numpy as np

## Data path handlers
TRAIN_TEST_SET_PATH = 'dataset/'
META_DATA_PATH = 'meta/'
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

import tensorflow as tf
from tensorflow import keras
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[1556]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy'
                 ,optimizer=optimizer
                 ,metrics=['accuracy'])
    return model

mlp_simple = keras.wrappers.scikit_learn.KerasClassifier(build_model)

early_stopping = keras.callbacks.EarlyStopping(patience=15,restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint(MODEL_PATH+"mlp_simple.h5",save_best_only=True)                                     

get_stats = pd.DataFrame([])
get_stats.to_csv(META_DATA_PATH+'mlp_error_analysis.csv')

class GetLossAnalysis(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        get_stats = pd.read_csv(META_DATA_PATH+'mlp_error_analysis.csv', index_col=0)
        
        if len(get_stats.columns) == 0:
            get_stats.columns = ['epoch', 'train_loss', 'test_loss' ]
            
        get_stats['epoch'] = epoch
        get_stats['train_loss'] = logs.get('loss')
        get_stats['test_loss'] = logs.get('val_loss')
        
        get_stats.to_csv(META_DATA_PATH+'mlp_error_analysis.csv')

loss_analisys_cb = GetLossAnalysis()

### Setting parameters range for hyperparameter optimzation
### Here we will use random search as the optimization method
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


params = {
    'n_hidden': np.arange(1,18),
    'n_neurons': np.arange(1,125),
    'learning_rate': reciprocal(3e-4, 3e-2),
    'input_shape': [X_train_vl.shape[1]]
}
rnd_search = RandomizedSearchCV(mlp_simple, params, n_iter=10, cv=3, n_jobs=1)
rnd_search.fit(X_train_vl, y_train_vl, epochs=100,validation_split=0.15,callbacks=[early_stopping, checkpoint_cb])
rnd_search.best_score_
best_parameters = pd.DataFrame(rnd_search.best_params_, index=['values'])
best_parameters.to_csv(MODEL_PATH+'mlp_simple_bp.csv')

rnd_search.best_params_
pd.read_csv(MODEL_PATH+'mlp_simple_bp.csv', usecols=['learning_rate']).iloc[0]

best = rnd_search.best_estimator_.model
best.fit(X_train, y_train, epochs=200,
         validation_data=(X_test, y_test),
         callbacks=[early_stopping])