# encoding: utf-8
import numpy as np
#from numpy.random import RandomState
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
 
#create data

np.random.seed(1337)
rdm = np.random.RandomState(1)

dataset_size = 300

X = rdm.uniform(1, 300, (dataset_size, 2))
# print(type(X))

np.random.shuffle(X)

Y = [(x1 + x2) for (x1, x2) in X]
#print(Y)

#plt.scatter(X,Y)
#plt.show()
X_train,Y_train = X[:256],Y[:256]
print(X_train)
print(Y_train)

# build a model from the 1st layer to the last layer
model = Sequential()
model.add(Dense(4, activation='relu', input_dim=2))
model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
 
#choose loss function and optimizing method
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.000001))
 
print("Training.....")
model.fit(X_train, Y_train, epochs=2000, batch_size=32)

'''
for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)
    if step%100==0:
        print("train cost",cost)
'''
print ("Testing.....")

X_test,Y_test = X[256:],Y[256:]
cost = model.evaluate(X_test,Y_test,batch_size=40)
print ("test cost:",cost)
W,b = model.layers[0].get_weights()
print ("weight=",W,"bias=",b)
#plotting the prediction
Y_pred = model.predict(X_test)
print(X_test)
print("Y_pred===============================================================")
print(Y_pred)
print("Y_test===============================================================")
print(Y_test)
