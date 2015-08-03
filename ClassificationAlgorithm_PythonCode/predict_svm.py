import pandas as pd
import numpy as np
import random
import json

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
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

classifier = LinearSVC(C=0.01)
classifier.fit(X_train,y_train)
#print classifier.coef_
#print classifier.intercept_
coeff_df = pd.DataFrame(classifier.coef_)
coeff_dict = coeff_df.T.to_dict()

model_dict={}
model_dict['coeff'] = coeff_dict
model_dict['intercept'] = classifier.intercept_.tolist()

with open('linearsvc_model.json', 'w') as fp:
    out_str = json.dumps(model_dict,indent=2,ensure_ascii=False,sort_keys=True).encode('utf8')
    fp.write(out_str)

y_pred = classifier.predict(X_validation)
print("Validation Error: %s"% ( 1.0-accuracy_score(y_validation, y_pred)))
print confusion_matrix(y_validation, y_pred, labels=np.arange(0,10))


#classifier = SVC(C=500.0,kernel='poly',degree=2)
#classifier.fit(X_train,y_train)
#y_pred = classifier.predict(X_validation)
#print("Validation Error: %s"% ( 1.0-accuracy_score(y_validation, y_pred)))
#print confusion_matrix(y_validation, y_pred, labels=np.arange(0,10))
