import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import ConvLSTM2D, Conv2DTranspose, Input, Layer
from data_generator import SheetImageGenerator


class Conv2DTransposePlus(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv2d_transpose = Conv2DTranspose(filters, kernel_size)

    def call(self, inputs, *args, **kwargs):
        return tf.map_fn(self.conv2d_transpose, inputs)


input_size = [100, 100, 3]
input_layer = Input((3, *input_size))
x = Conv2DTransposePlus(8, 2)(input_layer)
x = Conv2DTransposePlus(16, 2)(x)
x = Conv2DTransposePlus(16, 3)(x)
x = ConvLSTM2D(filters=8, kernel_size=3, return_sequences=True)(x)
x = ConvLSTM2D(filters=3, kernel_size=3, activation='sigmoid')(x)
model = Model(input_layer, x)
model.compile(optimizer='adam', loss='mse')
model.summary()

# model.fit(
#     SheetImageGenerator('data2', batch_size=32, piece_size=input_size),
#     epochs=3,
#     steps_per_epoch=1000
# )
# model.save('models/model200_200')
