import numpy as np
from keras.models import Sequential
from keras.layers import Dense
num_samples = 100000
X_train = np.random.rand(num_samples, 2)
y_train = X_train[:, 0] + X_train[:, 1]
model = Sequential()
model.add(Dense(8, input_shape=(2,), activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
batch_size = 100
epochs = 100
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
test_input = np.array([[1, 2], [0.3, 0.4]])
predicted_sum = model.predict(test_input)
print("Predicted sums:")
print(predicted_sum)
