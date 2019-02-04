# Modified code based on user Akihiko solution: https://stackoverflow.com/questions/50677544/reflection-padding-conv2d

import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec

from keras.models import Model
from keras.layers import Input, Conv2D


class ReflectionPadding2D(Layer):
    def __init__(self, padding, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3]
        )
        return shape

    def call(self, x, mask=None):
        width_pad, height_pad = self.padding
        return tf.pad(
            x,
            [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]],
            'REFLECT'
        )


if __name__ == '__main__':
    # Example of use
    nn_input = Input((128, 128, 3))
    reflect_pad = ReflectionPadding2D(padding=(3, 3))(nn_input)
    conv2d = Conv2D(32, kernel_size=7, strides=1, padding="valid")(reflect_pad)
    model = Model(nn_input, conv2d)
    model.summary()
