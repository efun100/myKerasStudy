from keras.models import load_model
import numpy as np

rdm = np.random.RandomState(1)
 
model = load_model('division_model.h5')

X = rdm.uniform(1, 1000, (10, 2))
print(X)
Y = [(x1 / x2) for (x1, x2) in X]

Y_pred = model.predict(X)
print(Y)
print(Y_pred)
