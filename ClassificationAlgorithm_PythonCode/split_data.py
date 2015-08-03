import pandas as pd
import random

print "loading the data"
mnist_data = pd.read_csv('data/train.csv', sep=',') 
print mnist_data.shape 

print mnist_data['label'].value_counts(normalize=True)
mnist_data.describe()

#generate training data
train_size = int(0.7*mnist_data.shape[0])
train_idx =  random.sample(mnist_data.index,train_size)
train_data = mnist_data.ix[train_idx]

mnist_data_30 = mnist_data.drop(train_idx)
test_size = int(0.5*mnist_data_30.shape[0])
test_idx = random.sample(mnist_data_30.index, test_size)
test_data = mnist_data_30.ix[test_idx]

validation_data = mnist_data_30.drop(test_idx)

train_data.to_csv('data/train_data.csv',index=False)
test_data.to_csv('data/test_data.csv',index=False)
validation_data.to_csv('data/validation_data.csv',index=False)
