import pandas as pd
import random
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def rescale_data(df):
    df = df.drop('label',1)
    df_scaled = df.apply(lambda x: x/255.0)
    return df_scaled.as_matrix()

print 'loading train/test data'
train_data = pd.read_csv('data/train_data.csv', sep=',') 
test_data  = pd.read_csv('data/test_data.csv', sep=',') 

X_train = rescale_data(train_data)
y_train = train_data['label'].values 

X_test = rescale_data(test_data) 
y_test = test_data['label'].values 

C_param_list = [1e-6,1e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,1]
print 'training linear kernel'
for C_param in  C_param_list:
    classifier = LinearSVC(C=C_param)
    classifier.fit(X_train,y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    print("C/Train Error/Test Error:%0.8f, %s, %s"% (C_param, 1.0-accuracy_score(y_train, y_train_pred), 1.0-accuracy_score(y_test, y_test_pred)))

C_param_list = [1e-1,1,5,10,20,50,100,200,300,500]
print 'training poly kernel (degree=2)'
for C_param in  C_param_list:
    classifier = SVC(C=C_param,kernel='poly',degree=2)
    classifier.fit(X_train,y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    print("C/Train Error/Test Error:%0.8f, %s, %s"% (C_param, 1.0-accuracy_score(y_train, y_train_pred), 1.0-accuracy_score(y_test, y_test_pred)))
