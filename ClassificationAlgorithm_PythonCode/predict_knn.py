import pandas as pd
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def rescale_data(df):
    df = df.drop('label',1)
    df_scaled = df.apply(lambda x: x/255.0)
    return df_scaled.as_matrix()

print 'loading train/validation data'
train_data = pd.read_csv('data/train_data.csv', sep=',') 
validation_data = pd.read_csv('data/validation_data.csv', sep=',') 

X_train = rescale_data(train_data)
y_train = train_data['label'].values 

X_validation = rescale_data(validation_data) 
y_validation = validation_data['label'].values 

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_validation)
print("Validation Error: %s"% ( 1.0-accuracy_score(y_validation, y_pred)))
print confusion_matrix(y_validation, y_pred, labels=np.arange(0,10))
