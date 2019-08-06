from tensorflow.python.keras import layers, Sequential
from tensorflow.python.keras.utils import plot_model

conv_filter = (3, 3)
conv_strides = (1, 1)
conv_padding = 'SAME'


def add_vgg_conv_pool_stack(num_conv, filters, model):
    """

    :param num_conv: Number of convolution layer stacks
    :param filters: number of filters
    :param model: keras/ tensroflow 2.0 model
    :return:
    """
    for i in range(num_conv):
        model.add(
            layers.Conv2D(filters=filters, kernel_size=conv_filter, activation='relu',
                          strides=conv_strides,
                          padding=conv_padding))
    model.add(layers.MaxPool2D(pool_size=pool_size, strides=pool_stride))


pool_size = (2, 2)
pool_stride = (2, 2)

# There are 16 convolutions and hence the name vgg16
model = Sequential()
model.add(layers.InputLayer(input_shape=(224, 224, 3)))

# Conv 64 x 2 and max pool
add_vgg_conv_pool_stack(filters=64, num_conv=2, model=model)

# Conv 128 x 2 and max pool
add_vgg_conv_pool_stack(filters=128, num_conv=2, model=model)

# Conv 256 x 3 and maxpool
add_vgg_conv_pool_stack(filters=256, num_conv=3, model=model)

# Conv 512 x 3 and maxpool
add_vgg_conv_pool_stack(filters=512, num_conv=3, model=model)

# Conv 512 x 3 and maxpool
add_vgg_conv_pool_stack(filters=512, num_conv=3, model=model)

model.add(layers.Flatten())

model.add(layers.Dense(units=4096, activation='relu'))
#
model.add(layers.Dense(units=4096, activation='relu'))
#
model.add(layers.Dense(units=1000, activation='softmax'))

plot_model(model, to_file=f'{__file__.split(".py")[0]}.png', show_layer_names=True, show_shapes=True)
