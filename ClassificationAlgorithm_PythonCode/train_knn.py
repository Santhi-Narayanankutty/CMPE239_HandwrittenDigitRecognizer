import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
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

#nn_params = [1,2,3,4,5,10,25]
nn_params = [1,2,3,4]

print 'training KNN'
for nn in  nn_params:
    classifier = KNeighborsClassifier(n_neighbors=nn)
    classifier.fit(X_train,y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    print("NN/Train Error/Validation Error:%d, %s, %s"% (nn, 1.0-accuracy_score(y_train, y_train_pred), 1.0-accuracy_score(y_test, y_test_pred)))
