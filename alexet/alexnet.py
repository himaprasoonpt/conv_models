from tensorflow.python.keras import layers, Sequential
from tensorflow.python.keras.utils import plot_model

# Local Response Normalization is not used

model = Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), activation='relu', input_shape=(227, 227, 3), strides=(4, 4)))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding="SAME"))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding="SAME"))
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding="SAME"))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding="SAME"))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(units=4096, activation='relu'))
#
model.add(layers.Dense(units=4096, activation='relu'))
#
model.add(layers.Dense(units=1000, activation='softmax'))

plot_model(model, to_file=f'{__file__.split(".py")[0]}.png', show_layer_names=True, show_shapes=True)
