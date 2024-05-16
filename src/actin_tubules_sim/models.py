import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import constant as const
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Conv3D,
    Input,
    Lambda,
    Layer,
    LeakyReLU,
    UpSampling2D,
    add,
    concatenate,
    multiply,
)
from tensorflow.keras.models import Model


class NoiseSuppressionModule(Layer):

    def __init__(self, init_cutoff_freq=4.1, dxy=0.0926, init_slop=100):
        super().__init__()
        self.cutoff_freq = self.add_weight(
            shape=(1,),
            initializer=const(init_cutoff_freq),
            trainable=True,
            name="cutoff_freq",
        )
        self.slop = self.add_weight(
            shape=(1,),
            initializer=const(init_slop),
            trainable=True,
            name="slop",
        )
        self.dxy = tf.Variable(initial_value=dxy, trainable=False, name="dxy")

    def call(self, inputs):
        bs, ny, nx, nz, ch = inputs.get_shape().as_list()
        ny = tf.cast(ny, tf.float32)
        nx = tf.cast(nx, tf.float32)
        dkx = tf.divide(1, tf.multiply(nx, self.dxy))
        dky = tf.divide(1, tf.multiply(ny, self.dxy))

        y = tf.multiply(tf.cast(tf.range(-ny // 2, ny // 2), tf.float32), dky)
        x = tf.multiply(tf.cast(tf.range(-nx // 2, nx // 2), tf.float32), dkx)
        [X, Y] = tf.meshgrid(x, y)
        rdist = tf.sqrt(tf.square(X) + tf.square(Y))

        otf_mask = tf.sigmoid(tf.multiply(self.cutoff_freq - rdist, self.slop))
        otf_mask = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(otf_mask, 0), 0), 0
        )
        otf_mask = tf.tile(otf_mask, (1, nz, ch, 1, 1))
        otf_mask = tf.complex(otf_mask, tf.zeros_like(otf_mask))

        inputs = tf.complex(inputs, tf.zeros_like(inputs))
        inputs = tf.transpose(inputs, [0, 3, 4, 1, 2])
        fft_feature = tf.signal.fftshift(tf.signal.fft2d(inputs))
        output = tf.signal.ifft2d(
            tf.signal.fftshift(tf.multiply(otf_mask, fft_feature))
        )
        output = tf.transpose(output, [0, 3, 4, 1, 2])
        output = tf.math.real(output)

        return output


def GlobalAveragePooling(input):
    return tf.reduce_mean(input, axis=(1, 2, 3), keepdims=True)


def CALayer(input, channel, reduction=16):
    W = Lambda(GlobalAveragePooling)(input)
    W = Conv3D(
        channel // reduction, kernel_size=1, activation="relu", padding="same"
    )(W)
    W = Conv3D(channel, kernel_size=1, activation="sigmoid", padding="same")(W)
    mul = multiply([input, W])
    return mul


def RCAB3D(input, channel):
    conv = Conv3D(channel, kernel_size=3, padding="same")(input)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(channel, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    att = CALayer(conv, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup(input, channel, n_RCAB=5):
    conv = input
    for _ in range(n_RCAB):
        conv = RCAB3D(conv, channel)
    return conv


def RCAN3D(input_shape, n_ResGroup=4, n_RCAB=5):

    inputs = Input(input_shape)
    conv = Conv3D(64, kernel_size=3, padding="same")(inputs)
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)

    conv = Conv3D(256, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(input_shape[3], kernel_size=3, padding="same")(conv)
    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=inputs, outputs=output)

    return model


def RCANNSM3D(input_shape, n_ResGroup=4, n_RCAB=5):
    inputs = Input(input_shape)
    conv_input = Conv3D(64, kernel_size=3, padding="same")(inputs)
    NSM = NoiseSuppressionModule()
    inputs_ns = NSM(inputs)
    conv = Conv3D(64, kernel_size=3, padding="same")(inputs_ns)
    conv = conv + conv_input
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)

    conv = Conv3D(256, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(input_shape[3], kernel_size=3, padding="same")(conv)
    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=inputs, outputs=output)

    return model


def global_average_pooling(input):
    return tf.reduce_mean(input, axis=(1, 2), keepdims=True)


def FCALayer(input, channel, reduction=16):
    absfft1 = Lambda(fft2)(input)
    absfft1 = Lambda(fftshift)(absfft1)
    absfft1 = tf.abs(absfft1, name="absfft1")
    absfft1 = tf.cast(absfft1, dtype=tf.float32)
    absfft2 = Conv2D(
        channel, kernel_size=3, activation="relu", padding="same"
    )(absfft1)
    W = Lambda(global_average_pooling)(absfft2)
    W = Conv2D(
        channel // reduction, kernel_size=1, activation="relu", padding="same"
    )(W)
    W = Conv2D(channel, kernel_size=1, activation="sigmoid", padding="same")(W)
    mul = multiply([input, W])
    return mul


def FCAB(input, channel):
    conv = Conv2D(channel, kernel_size=3, padding="same")(input)
    conv = Lambda(gelu)(conv)
    conv = Conv2D(channel, kernel_size=3, padding="same")(conv)
    conv = Lambda(gelu)(conv)
    att = FCALayer(conv, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup(input, channel):
    conv = input
    n_RCAB = 4
    for _ in range(n_RCAB):
        conv = FCAB(conv, channel)
    conv = add([conv, input])
    return conv


def DFCAN(input_shape, scale=2):
    inputs = Input(input_shape)
    print(inputs.shape)
    conv = Conv2D(64, kernel_size=3, padding="same")(inputs)
    conv = Lambda(gelu)(conv)
    n_ResGroup = 4
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, channel=64)
    conv = Conv2D(64 * (scale**2), kernel_size=3, padding="same")(conv)
    conv = Lambda(gelu)(conv)

    upsampled = Lambda(pixelshuffle, arguments={"scale": scale})(conv)
    conv = Conv2D(1, kernel_size=3, padding="same")(upsampled)
    output = Activation("sigmoid")(conv)
    model = Model(inputs=inputs, outputs=output)
    return model


def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf


def pixelshuffle(layer_in, scale):
    print(f"{layer_in},{scale}")
    return tf.nn.depth_to_space(
        layer_in, block_size=scale
    )  # here I changes :  block_size=scale :  to :  block_size=2*scale  :


class NSM(Layer):

    def __init__(self, init_cutoff_freq, init_slop=100, dxy=0.0626, **kwargs):
        super().__init__(**kwargs)
        self.cutoff_freq = self.add_weight(
            shape=(1,),
            initializer=const(init_cutoff_freq),
            trainable=True,
            name="cutoff_freq",
        )
        self.slop = self.add_weight(
            shape=(1,),
            initializer=const(init_slop),
            trainable=True,
            name="slop",
        )
        self.dxy = tf.Variable(initial_value=dxy, trainable=False, name="dxy")

    def call(self, inputs, **kwargs):
        bs, ny, nx, ch = inputs.get_shape().as_list()
        nx = tf.cast(nx, tf.float32)
        ny = tf.cast(ny, tf.float32)
        dkx = tf.divide(1, (tf.multiply(nx, self.dxy)))
        dky = tf.divide(1, (tf.multiply(ny, self.dxy)))

        y = tf.multiply(
            tf.cast(tf.range(-ny // 2, ny // 2), dtype=tf.float32), dky
        )
        x = tf.multiply(
            tf.cast(tf.range(-nx // 2, nx // 2), dtype=tf.float32), dkx
        )
        [map_x, map_y] = tf.meshgrid(x, y)
        rdist = tf.sqrt(tf.square(map_x) + tf.square(map_y))

        otf_mask = tf.sigmoid(tf.multiply(self.cutoff_freq - rdist, self.slop))
        otf_mask = tf.expand_dims(tf.expand_dims(otf_mask, 0), -1)
        otf_mask = tf.tile(otf_mask, (1, 1, 1, ch))

        otf_mask = tf.complex(otf_mask, tf.zeros_like(otf_mask))
        fft_feature = fftshift(fft2(inputs))
        output = ifft2(fftshift(tf.multiply(otf_mask, fft_feature)))

        return tf.math.real(output)


def ifft2(input):
    temp = K.permute_dimensions(input, (0, 3, 1, 2))
    ifft = tf.signal.ifft2d(temp)
    output = K.permute_dimensions(ifft, (0, 2, 3, 1))
    return output


def fft2(input):
    temp = K.permute_dimensions(input, (0, 3, 1, 2))
    fft = tf.signal.fft2d(tf.complex(temp, tf.zeros_like(temp)))
    output = K.permute_dimensions(fft, (0, 2, 3, 1))
    return output


def fftshift(input):
    bs, h, w, ch = input.get_shape().as_list()
    fs11 = input[:, -h // 2 : h, -w // 2 : w, :]
    fs12 = input[:, -h // 2 : h, 0 : w // 2, :]
    fs21 = input[:, 0 : h // 2, -w // 2 : w, :]
    fs22 = input[:, 0 : h // 2, 0 : w // 2, :]
    output = tf.concat(
        [tf.concat([fs11, fs21], axis=1), tf.concat([fs12, fs22], axis=1)],
        axis=2,
    )
    return output


def CALayer2D(input, input_height, input_width, channel, reduction=16):
    W = AveragePooling2D(pool_size=(input_height, input_width))(input)
    W = Conv2D(
        channel // reduction, kernel_size=1, activation="relu", padding="same"
    )(W)
    W = Conv2D(channel, kernel_size=1, activation="sigmoid", padding="same")(W)
    W = UpSampling2D(size=(input_height, input_width))(W)
    mul = multiply([input, W])
    return mul


def RCAB2D(input, input_height, input_width, channel):
    conv = Conv2D(channel, kernel_size=3, padding="same")(input)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv2D(channel, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    att = CALayer2D(conv, input_height, input_width, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup2D(input, input_height, input_width, channel):
    conv = input
    n_RCAB = 5
    for _ in range(n_RCAB):
        conv = RCAB2D(conv, input_height, input_width, channel)
    output = add([conv, input])
    return output


def DenoiserNSM(
    input_shape, n_rg=(2, 5, 5), init_cutoff_freq=4.95, init_slop=100
):

    inputs1 = Input(input_shape)
    inputs2 = Input(input_shape)
    oa = NSM(
        init_cutoff_freq=init_cutoff_freq, init_slop=init_slop, dxy=0.0626
    )(inputs2)
    conv1 = Conv2D(32, kernel_size=3, padding="same")(oa)
    conv2 = Conv2D(32, kernel_size=3, padding="same")(inputs2)
    inputs2_oa = concatenate([conv1, conv2], axis=3)

    # --------------------------------------------------------------------------------
    #                      extract features of generated image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding="same")(inputs1)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[0]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding="same")(conv)
    conv1 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                      extract features of noisy image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding="same")(inputs2_oa)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[1]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding="same")(conv)
    conv2 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                              merge features
    # --------------------------------------------------------------------------------
    conct = add([conv1, conv2])
    conct = Conv2D(64, kernel_size=3, padding="same")(conct)
    conct = LeakyReLU(alpha=0.2)(conct)
    conv = conct

    for _ in range(n_rg[2]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conct])

    conv = Conv2D(256, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    CA = CALayer2D(conv, input_shape[0], input_shape[1], 256, reduction=16)
    conv = Conv2D(input_shape[2], kernel_size=3, padding="same")(CA)

    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=[inputs1, inputs2], outputs=output)
    return model


def Denoiser(input_shape, n_rg=(2, 5, 5)):

    inputs1 = Input(input_shape)
    inputs2 = Input(input_shape)
    # --------------------------------------------------------------------------------
    #                      extract features of generated image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding="same")(inputs1)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[0]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding="same")(conv)
    conv1 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                      extract features of noisy image
    # --------------------------------------------------------------------------------
    conv0 = Conv2D(64, kernel_size=3, padding="same")(inputs2)
    conv = LeakyReLU(alpha=0.2)(conv0)
    for _ in range(n_rg[1]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conv0])
    conv = Conv2D(64, kernel_size=3, padding="same")(conv)
    conv2 = LeakyReLU(alpha=0.2)(conv)

    # --------------------------------------------------------------------------------
    #                              merge features
    # --------------------------------------------------------------------------------
    weight1 = Lambda(lambda x: x * 1)
    weight2 = Lambda(lambda x: x * 1)
    conv1 = weight1(conv1)
    conv2 = weight2(conv2)

    conct = add([conv1, conv2])
    conct = Conv2D(64, kernel_size=3, padding="same")(conct)
    conct = LeakyReLU(alpha=0.2)(conct)
    conv = conct

    for _ in range(n_rg[2]):
        conv = ResidualGroup2D(conv, input_shape[0], input_shape[1], 64)
    conv = add([conv, conct])

    conv = Conv2D(256, kernel_size=3, padding="same")(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    CA = CALayer2D(conv, input_shape[0], input_shape[1], 256, reduction=16)
    conv = Conv2D(input_shape[2], kernel_size=3, padding="same")(CA)

    output = LeakyReLU(alpha=0.2)(conv)

    model = Model(inputs=[inputs1, inputs2], outputs=output)
    return model
