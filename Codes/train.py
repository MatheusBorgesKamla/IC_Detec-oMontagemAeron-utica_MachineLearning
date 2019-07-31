import numpy as np

# ---- SVM ----
def train_SVM(X_train,X_test,y_train,epochs):
    #Implementando a SVM
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', probability=True)
    classifier.fit(X_train, y_train)
    #Prevendo os resultados de teste
    y_pred = classifier.predict(X_test)
    svm_predict = classifier.predict_proba(X_test)

    return y_pred