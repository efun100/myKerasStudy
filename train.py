# encoding: utf-8
import numpy as np
#from numpy.random import RandomState
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ReduceLROnPlateau
 
#create data

np.random.seed(1337)
rdm = np.random.RandomState(1)

dataset_size = 8300

X = rdm.uniform(1, 1000, (dataset_size, 2))
# print(type(X))

np.random.shuffle(X)

Y = [(x1 / x2) for (x1, x2) in X]
#print(Y)

#plt.scatter(X,Y)
#plt.show()
X_train,Y_train = X[:8192],Y[:8192]
print(X_train)
print(Y_train)
print(min(Y_train))
print(max(Y_train))

# build a model from the 1st layer to the last layer
model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(LeakyReLU())

model.add(Dense(16))
model.add(LeakyReLU())

model.add(Dense(32))
model.add(LeakyReLU())

model.add(Dense(64))
model.add(LeakyReLU())

model.add(Dense(32))
model.add(LeakyReLU())

model.add(Dense(16))
model.add(LeakyReLU())

model.add(Dense(8))
model.add(LeakyReLU())
#model.add(Dense(4, activation='relu'))
#model.add(Dense(4, activation='relu'))
model.add(Dense(1))
 
#choose loss function and optimizing method
model.compile(loss='mean_absolute_percentage_error', optimizer=keras.optimizers.SGD(lr=0.000001, momentum=0.9))

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=200, mode='auto')
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=500, verbose=0, mode='auto')
 
print("Training.....")
model.fit(X_train, Y_train, epochs=4000, batch_size=64)

model.save("division_model.h5")

'''
for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)
    if step%100==0:
        print("train cost",cost)
'''
print ("Testing.....")

X_test,Y_test = X[8192:],Y[8192:]
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
