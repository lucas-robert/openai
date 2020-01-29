from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(5, input_shape=(3 ,),activation='relu'),
    Dense(2, activation='softmax'),
])
