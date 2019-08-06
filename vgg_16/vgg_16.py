from tensorflow.python.keras import layers, Sequential
from tensorflow.python.keras.utils import plot_model

conv_filter = (3, 3)
conv_strides = (1, 1)
conv_padding = 'SAME'

pool_size = (2, 2)
pool_stride = (2, 2)
# There are 16 convolutions and hence the name vgg16
model = Sequential()
model.add(layers.InputLayer(input_shape=(224, 224, 3)))
# Conv 64 x 2 and max pool
model.add(
    layers.Conv2D(filters=64, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(
    layers.Conv2D(filters=64, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))

model.add(layers.MaxPool2D(pool_size=pool_size, strides=pool_stride))

# Conv 128 x 2 and max pool
model.add(
    layers.Conv2D(filters=128, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(
    layers.Conv2D(filters=128, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(layers.MaxPool2D(pool_size=pool_size, strides=pool_stride))

# Conv 256 x 3 and maxpool
model.add(
    layers.Conv2D(filters=256, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(
    layers.Conv2D(filters=256, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(
    layers.Conv2D(filters=256, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(layers.MaxPool2D(pool_size=pool_size, strides=pool_stride))

model.add(
    layers.Conv2D(filters=256, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))

# Conv 512 x 3 and maxpool
model.add(
    layers.Conv2D(filters=512, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(
    layers.Conv2D(filters=512, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(
    layers.Conv2D(filters=512, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(layers.MaxPool2D(pool_size=pool_size, strides=pool_stride))

# Conv 512 x 3 and maxpool
model.add(
    layers.Conv2D(filters=512, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(
    layers.Conv2D(filters=512, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(
    layers.Conv2D(filters=512, kernel_size=conv_filter, activation='relu',
                  strides=conv_strides,
                  padding=conv_padding))
model.add(layers.MaxPool2D(pool_size=pool_size, strides=pool_stride))

model.add(layers.Flatten())

model.add(layers.Dense(units=4096, activation='relu'))
#
model.add(layers.Dense(units=4096, activation='relu'))
#
model.add(layers.Dense(units=1000, activation='softmax'))

plot_model(model, to_file=f'{__file__.split(".py")[0]}.png', show_layer_names=True, show_shapes=True)
