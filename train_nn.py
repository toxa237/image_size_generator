import tensorflow as tf
from keras.models import Model
from keras.layers import ConvLSTM2D, Conv2DTranspose, Input, Layer
from data_generator import SheetImageGenerator


class Conv2DTransposePlus(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv2d_transpose = Conv2DTranspose(filters, kernel_size)

    def call(self, inputs, *args, **kwargs):
        return tf.map_fn(self.conv2d_transpose, inputs)


input_size = [50, 50, 3]
input_layer = Input((3, *input_size))
x = Conv2DTransposePlus(8, 2)(input_layer)
x = Conv2DTransposePlus(16, 2)(x)
x = ConvLSTM2D(filters=8, kernel_size=2, return_sequences=True)(x)
x = ConvLSTM2D(filters=3, kernel_size=2, activation='sigmoid')(x)
model = Model(input_layer, x)
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(SheetImageGenerator('data2', batch_size=32, piece_size=input_size), epochs=1)
model.save('models/model3')



