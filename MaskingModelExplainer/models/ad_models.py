from tensorflow import keras
import tensorflow as tf


def define_ad_model(in_shape):
    inputs = keras.layers.Input(in_shape)
    l = keras.layers.Dense(128, activation='relu')(inputs)
    #l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.3)(l)
    l = keras.layers.Dense(64, activation='relu')(l)
    #l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.3)(l)
    outputs = keras.layers.Dense(1, activation='sigmoid')(l)

    return keras.Model(inputs=inputs, outputs=outputs)

def define_ad_dummy(in_shape):
    inputs = keras.layers.Input(in_shape)
    #l = keras.layers.Dropout(0.4)(inputs)
    l = keras.layers.Dense(32, activation='relu')(inputs)
    l = keras.layers.BatchNormalization()(l)
    #l = keras.layers.Dropout(0.4)(l)
    l = keras.layers.Dense(16, activation='relu')(l)
    l = keras.layers.BatchNormalization()(l)
    #l = keras.layers.Dropout(0.3)(l)
    #l = keras.layers.Dropout(0.5)(inputs)
    #l = keras.layers.Dense(4, activation='relu', kernel_regularizer=keras.regularizers.L1())(l)
    #l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.4)(l)
    outputs = keras.layers.Dense(2, activation='softmax')(l)

    return keras.Model(inputs=inputs, outputs=outputs, name='class_out')

def model_with_fs(in_shape):
    inputs = keras.layers.Input(in_shape)
    l = keras.layers.Dense(32, activation='relu')(inputs)
    l = keras.layers.Dense(8, activation='relu')(l)
    l = keras.layers.Dense(32, activation='relu')(l)
    l = keras.layers.Dense(in_shape, activation='sigmoid', name='sel_inputs')(l)
    sel_input = keras.layers.Multiply()([inputs, l])

    ad_model = define_ad_dummy(in_shape)
    classification = ad_model(sel_input)

    return keras.Model(inputs=inputs, outputs=[classification, l])


def define_ad_model_complex(in_shape):
    inputs = keras.layers.Input(in_shape)
    l = keras.layers.Dense(128, activation='relu')(inputs)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.3)(l)
    l = keras.layers.Dense(64, activation='relu')(l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.3)(l)
    l = keras.layers.Dense(32, activation='relu')(l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.3)(l)
    outputs = keras.layers.Dense(1, activation='sigmoid')(l)

    return keras.Model(inputs=inputs, outputs=outputs)


def define_ad_model_synthetic_dataset(in_shape, bias_init):
    inputs = keras.layers.Input(in_shape)
    l = keras.layers.Dense(64, activation='relu')(inputs)
    #l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.4)(l)
    l = keras.layers.Dense(128, activation='relu')(l)
   # l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.4)(l)
    l = keras.layers.Dense(256, activation='relu')(l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.3)(l)
    # l = keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.L1())(inputs)
    # l = keras.layers.BatchNormalization()(l)
    # l = keras.layers.Dense(16, activation='relu')(l)
    # l = keras.layers.BatchNormalization()(l)
    # l = keras.layers.Dropout(0.3)(l)
    # l = keras.layers.Dense(16, activation='relu')(l)
    # l = keras.layers.BatchNormalization()(l)
    # l = keras.layers.Dropout(0.3)(l)
    outputs = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(bias_init))(l)

    return keras.Model(inputs=inputs, outputs=outputs)

def build_ae(in_shape):
    inputs = keras.layers.Input(in_shape)
    #l = keras.layers.Dense(128, activation='relu')(inputs)
    #l = keras.layers.Dense(32, activation='relu')(l)
    l = keras.layers.Dense(2)(inputs)
    #l = keras.layers.Dense(32, activation='relu')(l)
    #l = keras.layers.Dense(128, activation='relu')(l)
    outputs = keras.layers.Dense(in_shape)(l)
    return keras.Model(inputs=inputs, outputs=outputs)

def build_vae(in_shape):
    encoder_inputs = keras.layers.Input(in_shape)
    l = keras.layers.Dense(256, activation='relu')(encoder_inputs)  # 128
    l = keras.layers.Dense(64, activation='relu')(l)        # 32
    l = keras.layers.Dense(32, activation='relu')(l)
    z_mean = keras.layers.Dense(in_shape//2, name="z_mean")(l)
    z_log_var = keras.layers.Dense(in_shape//2, name="z_log_var")(l)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.layers.Input(in_shape//2)
    l = keras.layers.Dense(32, activation='relu')(latent_inputs)
    l = keras.layers.Dense(64, activation='relu')(l)
    l = keras.layers.Dense(256, activation='relu')(l)
    decoder_outputs = keras.layers.Dense(in_shape)(l)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return VAE(encoder, decoder)


# this class takes encoder and decoder models and
# define the complete variational autoencoder architecture
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=None, mask=None):
        _, _, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_sum((data - reconstruction)**2, axis=1)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss *= -0.5
            # beta =10
            total_loss = reconstruction_loss + 10 * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# this sampling layer is the bottleneck layer of variational autoencoder,
# it uses the output from two dense layers z_mean and z_log_var as input,
# convert them into normal distribution and pass them to the decoder layer
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon