import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, tanh


class SpectralNormalization(Constraint):
    def __init__(self, n_iter=5):
        # n_iter is a hyperparam
        self.n_iter = n_iter

    def call(self, input_weights):
        # reshape weights from conv layer into (HxW, C)
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))

        u = tf.random.normal((w.shape[0], 1))

        for _ in range(self.n_iter):
            v = tf.matmul(w, u, transpose_a=True)
            v /= tf.norm(v)

            u = tf.matmul(w, v)
            u /= tf.norm(u)

        spec_norm = tf.matmul(u, tf.matmul(w, v), transpose_a=True)

        return input_weights / spec_norm


class SelfAttention(Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()
    
    def build(self, input_shape):
        n, h, w, c = input_shape
        self.n_feats = h * w
        self.conv_theta = Conv2D(c // 8, 1,
                                 padding='same', 
                                 kernel_constraint=SpectralNormalization(), 
                                 name='Conv_Theta')

        self.conv_phi = Conv2D(c // 8, 1,
                               padding='same', 
                               kernel_constraint=SpectralNormalization(),
                               name='Conv_Phi')

        self.conv_g = Conv2D(c // 2, 1,
                             padding='same', 
                             kernel_constraint=SpectralNormalization(),
                             name='Conv_G')

        self.conv_attn_g = Conv2D(c, 1,
                                  padding='same', 
                                  kernel_constraint=SpectralNormalization(),
                                  name='Conv_AttnG')

        self.sigma = self.add_weight(shape=[1],
                                     initializer='zeros',
                                     trainable=True,
                                     name='sigma')

    def call(self, x):
        # theta => key
        # phi => query
        # g => query

        n, h, w, c = x.shape

        theta = self.conv_theta(x)
        theta = tf.reshape(theta, (-1, self.n_feats, theta.shape[-1]))
        
        phi = self.conv_phi(x)
        phi = tf.nn.max_pool2d(phi, ksize=2, strides=2, padding='VALID')
        phi = tf.reshape(phi, (-1, self.n_feats // 4, phi.shape[-1]))
        
        # generate attention map
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        g = self.conv_g(x)
        g = tf.nn.max_pool2d(g, ksize=2, strides=2, padding='VALID')
        g = tf.reshape(g, (-1, self.n_feats // 4, g.shape[-1]))

        # multiply attn map with feature maps
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, (-1, h, w, attn_g.shape[-1]))
        attn_g = self.conv_attn_g(attn_g)
        
        output = x + self.sigma * attn_g
        
        return output


class ConditionalBatchNorm(Layer):
    def __init__(self, n_class=2, decay_rate=0.999, eps=1e-7):
        super(ConditionalBatchNorm, self).__init__()
        self.n_class = n_class
        self.decay = decay_rate
        self.eps = 1e-5

    def build(self, input_shape):
        self.input_size = input_shape
        n, h, w, c = input_shape

        self.gamma = self.add_weight(
            shape=[self.n_class, c],
            initializer="ones",
            trainable=True,
            name="gamma"
        )

        self.beta = self.add_weight(
            shape=[self.n_class, c],
            initializer="zeros",
            trainable=True,
            name="beta"
        )

        self.moving_mean = self.add_weight(
            shape=[1, 1, 1, c],
            initializer="zeros",
            trainable=False,
            name="moving_mean"
        )

        self.moving_var = self.add_weight(
            shape=[1, 1, 1, c],
            initializer="ones",
            trainable=False,
            name="moving_var"
        )


    def call(self, x, labels, training=False):
        beta = tf.gather(self.beta, labels)
        beta = tf.expand_dims(beta, 1)
        gamma = tf.gather(self.gamma, labels)
        gamma = tf.expand_dims(gamma, 1)

        if training:
            mean, var = tf.nn.moments(x, axes=(0, 1, 2), keepdims=True)
            self.moving_mean.assign(self.decay * self.moving_mean + (1-self.decay)*mean)
            self.moving_var.assign(self.decay * self.moving_var + (1-self.decay)*var)
            output = tf.nn.batch_normalization(x, mean, var, beta, gamma, self.eps)
        else:
            output = tf.nn.batch_normalization(
                x,
                self.moving_mean,
                self.moving_var,
                beta,
                gamma,
                self.eps
            )

        return output


class ResBlockDown(Layer):
    def __init__(self, filters, downsample=True):
        super(ResBlockDown, self).__init__()
        self.filters = filters
        self.downsample = downsample


    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = Conv2D(self.filters, 3, padding="same", kernel_constraint=SpectralNormalization())

        self.conv_2 = Conv2D(self.filters, 3, padding="same", kernel_constraint=SpectralNormalization())

        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = Conv2D(self.filters, 1, padding="same", kernel_constraint=SpectralNormalization())

    def down(self, x):
        return tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

    def call(self, input_tensor):
        x = self.conv_1(input_tensor)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x, 0.2)

        if self.downsample:
            x = self.down(x)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = tf.nn.leaky_relu(skip, 0.2)

            if self.downsample:
                skip = self.down(skip)
        else:
            skip = input_tensor

        output = skip + x

        return output


class ResBlock(Layer):
    def __init__(self, filters, n_class):
        super(ResBlock, self).__init__(name=f"g_resblock_{filters}x{filters}")
        self.filters = filters
        self.n_class = n_class


    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = Conv2D(self.filters, 3, padding="same", name="conv2d_1", kernel_constraint=SpectralNormalization())

        self.conv_2 = Conv2D(self.filters, 3, padding="same", name="conv2d_2", kernel_constraint=SpectralNormalization())

        self.cbn_1 = ConditionalBatchNorm(self.n_class)
        self.cbn_2 = ConditionalBatchNorm(self.n_class)

        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = Conv2D(self.filters, 1, padding="same", name="conv2d_3", kernel_constraint=SpectralNormalization())
            self.cbn_3 = ConditionalBatchNorm(self.n_class)

    def call(self, input_tensor, labels):
        x = self.conv_1(input_tensor)
        x = self.cbn_1(x, labels)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        x = self.cbn_2(x, labels)
        x = tf.nn.leaky_relu(x, 0.2)

        if self.learned_skip:
            skip = self.conv_3(input_tensor)
            skip = self.cbn_3(skip, labels)
            skip = tf.nn.leaky_relu(skip, 0.2)
        else:
            skip = input_tensor

        output = skip + x
        return output


def build_generator(latent_dim, n_class):
    DIM = 64
    z = Input(shape=(latent_dim))
    labels = Input(shape=(1), dtype="int32")

    x = Dense(4 * 4 * 4 * DIM, kernel_constraint=SpectralNormalization())(z)
    x = Reshape((4, 4, 4 * DIM))(x)

    x = UpSampling2D((2, 2))(x)
    x = ResBlock(4 * DIM, n_class)(x, labels)

    x = UpSampling2D((2, 2))(x)
    x = ResBlock(2 * DIM, n_class)(x, labels)

    x = SelfAttention()(x)

    x = UpSampling2D((2, 2))(x)
    x = ResBlock(DIM, n_class)(x, labels)

    output_img = tanh(Conv2D(3, 3, padding="same", kernel_constraint=SpectralNormalization())(x))

    return Model([z, labels], output_img, name="generator")


def build_discriminator(n_class, img_shape):
    DIM = 64
    input_img = Input(shape=img_shape)
    input_labels = Input(shape=(1))

    embedding = Embedding(n_class, 4 * DIM)(input_labels)

    embedding = Flatten()(embedding)

    x = ResBlockDown(DIM)(input_img) # 64
    x = ResBlockDown(2 * DIM)(x) # 32
    x = SelfAttention()(x)
    x = ResBlockDown(4 * DIM)(x) # 16
    x = ResBlockDown(4 * DIM, False)(x) # 4
    x = tf.reduce_sum(x, (1, 2))

    embedded_x = tf.reduce_sum(x * embedding, axis=1, keepdims=True)

    output = Dense(1)(x)

    output += embedded_x

    return Model([input_img, input_labels], output, name="discriminator")


def hinge_loss_d(y, is_real):
    label = 1. if is_real else -1.
    loss = tf.keras.losses.Hinge()(y, label)
    return loss


def hinge_loss_g(y):
    return -tf.reduce_mean(y)


class SAGAN(Model):
    def __init__(self, discriminator, generator, image_shape, n_class, batch_size, z_dim=128):
        super(SAGAN, self).__init__()

        self.image_shape = image_shape
        self.n_class = n_class
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

        # Build models
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_opt, g_opt, d_loss_fn, g_loss_fn):
        super(SAGAN, self).compile()
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    
    def train_step(self, train_gen):
        # Train
        real_imgs, real_class_labels = (train_gen)
        batch_size = tf.shape(real_class_labels)[0]

        real_labels = 1
        fake_labels = 0

        z = tf.random.normal(shape=(batch_size, self.z_dim))
        fake_class_labels = real_class_labels

        with tf.GradientTape() as d_tape, \
             tf.GradientTape() as g_tape:

             # Forward pass
             fake_imgs = self.generator([z, fake_class_labels])
             pred_real = self.discriminator([real_imgs, real_class_labels])
             pred_fake = self.discriminator([fake_imgs, fake_class_labels])

             # discriminator losses
             loss_fake = self.d_loss_fn(pred_fake, False)
             loss_real = self.d_loss_fn(pred_real, True)

             # total loss
             d_loss = 0.5 * (loss_fake + loss_real)
             d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
             self.d_opt.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

             # generator loss
             g_loss = self.g_loss_fn(pred_fake)
             g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
             self.g_opt.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

             self.d_loss_metric.update_state(d_loss)
             self.g_loss_metric.update_state(g_loss)

             return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}


