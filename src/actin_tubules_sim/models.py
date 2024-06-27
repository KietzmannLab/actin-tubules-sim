import tensorflow as tf
import numpy as np
import numpy.fft as F
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
from .sim_fitting import cal_modamp, create_psf
import cv2 


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


class Train_RDL_Denoising(tf.keras.Model):
    def __init__(self, srmodel, denmodel, loss_fn, optimizer,  parameters, PSF = None):
        super(Train_RDL_Denoising, self).__init__()
        self.srmodel = srmodel
        self.denmodel = denmodel
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.PSF = PSF
        self.parameters = parameters 
        self.nphases = self.parameters['nphases']
        self.ndirs = self.parameters['ndirs']
        self.space = self.parameters['space']
        self.Ny = self.parameters['Ny']
        self.Nx = self.parameters['Nx']
        self.phase_space = 2 * np.pi / self.nphases
        self.scale = self.parameters['scale']
        self.dxy = self.parameters['dxy']
        self.sigma_x = self.parameters['sigma_x']
        self.sigma_y = self.parameters['sigma_y']
        self.dxy = self.parameters['dxy']
        [self.Nx_hr, self.Ny_hr] = [self.Nx* self.scale, self.Ny* self.scale] 
        [self.dx_hr, self.dy_hr] = [x / self.scale for x in [self.dxy, self.dxy]]

        xx = self.dx_hr * np.arange(-self.Nx_hr / 2, self.Nx_hr / 2, 1)
        yy = self.dy_hr * np.arange(-self.Ny_hr / 2, self.Ny_hr / 2, 1)
        [self.X, self.Y] = np.meshgrid(xx, yy)
        
        self.dkx = 1.0 / ( self.Nx *  self.dxy)
        self.dky = 1.0 / ( self.Ny * self.dxy)
        
        if self.PSF is None:
            
            self.PSF, self.OTF = create_psf(self.sigma_x, 
                        self.sigma_x,
                        self.Nx_hr, 
                        self.Ny_hr, 
                        self.dkx, 
                        self.dky)
        else:
            self.PSF /= np.sum(self.PSF)  
            self.OTF = abs(F.ifftshift(F.ifft2(self.PSF)))
            self.OTF /= np.sum(self.OTF)

    
    
    def _phase_computation(self, img_SR, modamp, cur_k0_angle, cur_k0):

            phase_list = -np.angle(modamp)
            img_gen = []
            for d in range(self.ndirs):
                alpha = cur_k0_angle[d]
                for i in range(self.nphases):
                    kxL = cur_k0[d] * np.pi * np.cos(alpha)
                    kyL = cur_k0[d] * np.pi * np.sin(alpha)
                    kxR = -cur_k0[d] * np.pi * np.cos(alpha)
                    kyR = -cur_k0[d] * np.pi * np.sin(alpha)
                    phOffset = phase_list[d] + i * self.phase_space
                    interBeam = np.exp(1j * (kxL * self.X + kyL * self.Y + phOffset)) + np.exp(1j * (kxR * self.X + kyR * self.Y))
                    pattern = np.square(np.abs(interBeam))
                    patterned_img_fft = F.fftshift(F.fft2(pattern * img_SR)) * self.OTF
                    modulated_img = np.abs(F.ifft2(F.ifftshift(patterned_img_fft)))
                    modulated_img = cv2.resize(modulated_img, (self.Ny, self.Nx))    
                    img_gen.append(modulated_img)
            
            img_gen = np.asarray(img_gen)
            
            
            return img_gen
    
    
    def _get_cur_k(self, image_gt):
        
        cur_k0, modamp = cal_modamp(np.array(image_gt).astype(np.float32), self.OTF, self.parameters)
        cur_k0_angle = np.array(np.arctan(cur_k0[:, 1] / cur_k0[:, 0]))
        cur_k0_angle[1:self.parameters['ndirs']] = cur_k0_angle[1:self.parameters['ndirs']] + np.pi
        cur_k0_angle = -(cur_k0_angle - np.pi/2)
        for nd in range(self.parameters['ndirs']):
            if np.abs(cur_k0_angle[nd] - self.parameters['k0angle_g'][nd]) > 0.05:
                cur_k0_angle[nd] = self.parameters['k0angle_g'][nd]
        cur_k0 = np.sqrt(np.sum(np.square(cur_k0), 1))
        given_k0 = 1 / self.parameters['space']
        cur_k0[np.abs(cur_k0 - given_k0) > 0.1] = given_k0
    
        return cur_k0, cur_k0_angle, modamp
    
    def _intensity_equilization(self, img_in, image_gt):
        # Compute the mean for the first nphases slice of img_in and image_gt
        mean_th_in = tf.reduce_mean(img_in[:self.nphases])
        mean_th_gt = tf.reduce_mean(image_gt[:self.nphases])
        
        # Reshape img_in to compute the mean for each direction
        img_in_reshaped = tf.reshape(img_in, (self.ndirs, self.nphases, img_in.shape[1], img_in.shape[2]))
        data_in_means = tf.reduce_mean(img_in_reshaped, axis=[1, 2, 3], keepdims=True)
        normalized_img_in = img_in * mean_th_in / tf.repeat(data_in_means, repeats=self.nphases, axis=0)
        
        # Reshape image_gt to compute the mean for each direction
        image_gt_reshaped = tf.reshape(image_gt, (self.ndirs, self.nphases, image_gt.shape[1], image_gt.shape[2]))
        data_gt_means = tf.reduce_mean(image_gt_reshaped, axis=[1, 2, 3], keepdims=True)
        normalized_image_gt = image_gt * mean_th_gt / tf.repeat(data_gt_means, repeats=self.nphases, axis=0)
        
        return normalized_img_in, normalized_image_gt
        
    
    def fit(self, data, data_val):
        x, y = data
        x_val, y_val = data_val
        input_height = x.shape[1]
        input_width = x.shape[2]
        batch_size = x.shape[0]
        channels = x.shape[-1]
        
        sr_y_predict = self.srmodel.predict(x)
        sr_y_predict = tf.squeeze(sr_y_predict, axis=-1) # Batch, Ny, Nx, 1 
        # Loop over each example in the batch
        for i in range(batch_size):
            # Get the current example
            img_in = x[i:i+1]  # Extract the i-th example from the batch
            img_SR = sr_y_predict[i:i+1]  # Extract the corresponding SR output
            image_gt = y[i:i+1]
            cur_k0, cur_k0_angle, modamp = self._get_cur_k(image_gt=image_gt)
            
            img_in, image_gt = self._intensity_equilization(img_in, image_gt)
            image_gen = self._phase_computation(img_SR, modamp, cur_k0_angle, cur_k0)
            print(img_in.shape, img_SR.shape, image_gt.shape, image_gen.shape)
            # Train denoising
            #with tf.GradientTape() as tape:
            #    y_pred = self.denmodel(img_in, training=True) 
            #    loss = self.loss_fn(y[i:i+1], y_pred)  

            #trainable_vars = self.denmodel.trainable_variables
            #gradients = tape.gradient(loss, trainable_vars)

            #self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            #self.compiled_metrics.update_state(y[i:i+1], y_pred)

        #return {m.name: m.result() for m in self.metrics}