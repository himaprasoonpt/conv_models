from tensorflow.python.keras import layers, Sequential
from tensorflow.python.keras.utils import plot_model

model = Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1), strides=(1, 1)))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', strides=(1, 1)))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation='softmax'))


plot_model(model, to_file=f'{__file__.split(".py")[0]}.png', show_layer_names=True, show_shapes=True)
