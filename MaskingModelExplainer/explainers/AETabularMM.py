from math import ceil

from explainers.MaskingModelExplainer import MaskingModelExplainer
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.keras.layers import Input, Dense, Add, Multiply, BatchNormalization
import neural_structured_learning as nsl
from utils.utils import focal_loss
import numpy as np


def relu_clipped(x, threshold=0., max_value=1.):
    return keras.backend.relu(x, threshold=threshold, max_value=max_value)


def custom_activation(x, beta=30):
    return 1 / (1 + tf.exp(- beta * x))

@tf.function
def euclidean(A, B):
    v = tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1)
    p1 = tf.reshape(tf.reduce_sum(v, axis=1), (-1, 1))
    v = tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1])
    p2 = tf.transpose(tf.reshape(tf.reduce_sum(v, axis=1), (-1, 1)))
    res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True))
    return (res)

@tf.function
def efficient_marahanobis(A, B, invS):
    '''
    A : tensor, N sample1 by N feat
    B : tensor, N sample2 by N feat
    S : tensor, N feat by N feat

    Output:

    marahanobis distance of each pair (A[i],B[j]) with inv variance S

    '''
    S_half = tf.linalg.cholesky(invS)
    A_star = tf.matmul(A, S_half)
    B_star = tf.matmul(B, S_half)

    res = euclidean(A_star, B_star)
    return (res)

class TabularMM(MaskingModelExplainer):

    def __init__(self, predict_fn, in_shape, data, optimizer=tf.keras.optimizers.RMSprop(), weights=None): #, minimums, maximums):
        super(TabularMM, self).__init__(predict_fn)
        self.MASKGEN, self.MASKAPPLY, self.PATCH = self.buildExplanator(predict_fn, in_shape)
        self.sigma = np.linalg.inv(np.cov(data.T)).astype(np.float32)
        self.optimizer = optimizer
        print(self.sigma.shape)
        if weights is None:
            self.weigths = np.ones(data.shape[0])
        else:
            self.weigths = weights

    def definePatch(self, in_shape):
        """
        Define the model that produce the patch from the original image
        :param in_shape: input shape
        :return:
        """

        # Mette insieme maskapply e maskgen
        img = keras.Input(shape=in_shape)
        MASKGEN = self.defineMaskGen(in_shape)
        MASKAPPLY = self.defineMaskApply(in_shape)
        mask = MASKGEN(img)
        patch = MASKAPPLY([img, mask[0], mask[1]])
        # Generatore patch - Applicatore della patch - Modello completo (gen+app)
        return MASKGEN, MASKAPPLY, keras.Model(inputs=img, outputs=patch)

    def defineMaskGen(self, in_shape):
        inputs = Input(in_shape)
        x0 = Dense(in_shape//2)(inputs)
        x0 = Dense(in_shape//4)(x0)
        x0 = Dense(in_shape//2)(x0)
        outputs = Dense(in_shape)(x0)

        x1 = Dense(in_shape//2, activation='relu')(inputs)
        x1 = Dense(in_shape//4, activation='relu')(x1)
        x1 = Dense(in_shape//2, activation='relu')(x1)
        outputs_c = Dense(in_shape, activation='sigmoid')(x1)
        #outputs_c = Lambda(lambda x: tf.round(x))(outputs_c)
        #outputs_c = ReLU(threshold=0.5)(outputs_c)

        return keras.Model(inputs=inputs, outputs=[outputs, outputs_c], name='MaskGen')

   #def defineMaskGen(self, in_shape):
   #    inputs = Input(in_shape)
   #    x0 = Dense(64)(inputs)
   #    #x0 = Dense(64)(x0)
   #    x0 = Dense(32)(x0)
   #    x0 = Dense(64)(x0)
   #    outputs = Dense(in_shape)(x0)

   #    x1 = Dense(64, activation='relu')(inputs)
   #    #x1 = Dense(64, activation='relu')(x1)
   #    #x1 = keras.layers.BatchNormalization()(x1)
   #    x1 = Dense(32, activation='relu')(x1)#, kernel_initializer=tf.keras.initializers.Constant(1e-3),
   #                    #bias_initializer=tf.keras.initializers.Zeros())(x1)
   #    #x1 = keras.layers.BatchNormalization()(x1)
   #    #x1 = Dense(32, activation='relu')(x1)
   #    x1 = Dense(64, activation='relu')(x1)
   #    #x1 = keras.layers.BatchNormalization()(x1)
   #    outputs_c = Dense(in_shape, activation='sigmoid')(x1)
   #    #outputs_c = Lambda(lambda x: tf.round(x))(outputs_c)
   #    #outputs_c = ReLU(threshold=0.5)(outputs_c)

   #    return keras.Model(inputs=inputs, outputs=[outputs, outputs_c], name='MaskGen')

    def defineMaskApply(self, in_shape):
        inputs = [Input(in_shape, name='input_img'), Input(in_shape, name='input_mask'),
                  Input(in_shape, name='input_choice')]  # Sample, Mask
        mid_output = Multiply()(inputs[1:])

        outputs = Add()([inputs[0], mid_output])

        #return keras.Model(inputs=inputs, outputs=[outputs, inputs[2]])
        return keras.Model(inputs=inputs, outputs=outputs)

    def fit_explanator(self, train_images_expl, train_labels_expl, epochs=1, verbose=0,
                       sample_weight=None, batch_size=32, loss_weights=[0.2, 1.5, 2.8], optimizer=tf.keras.optimizers.RMSprop()):
        binary_ce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        #sigma = np.var(train_images_expl, axis=0)
        #print('Sigma: ', self.sigma)
        if loss_weights is None:
            loss_weights = [1, .2, .4]

        data_a = train_images_expl[train_labels_expl == 1]
        data_n = train_images_expl[train_labels_expl == 0]
        for epoch in range(epochs):
            for i in range(ceil(train_images_expl[train_labels_expl == 1].shape[0] / batch_size)):
                #batch_n = data_n[i * batch_size:min((i + 1) * batch_size, data_n.shape[0])]
                batch_n = data_n
                #batch_a = np.repeat(data_a, batch_n.shape[0], axis=0)
                batch_a = data_a[i * batch_size:min((i + 1) * batch_size, data_a.shape[0])]
                # batch_label = train_labels_expl[
                #               i * batch_size:min((i + 1) * batch_size, train_images_expl[i * batch_size:].shape[0])]
                #
                # batch_a = batch[batch_label == 1]

                with tf.GradientTape() as tape:
                    masks, choose = self.MASKGEN(batch_a)
                    patches = self.MASKAPPLY([batch_a, masks, choose])
                    classification = self.model(patches)

                    # Reduce the number of samples
                    #ndim_loss = binary_ce(y_true=np.zeros_like(choose), y_pred=choose)
                    ndim_loss = tf.reduce_mean(choose, axis=1)
                    dim_distance = 0.0

                    # Classification error
                    cross_entropy = binary_ce(y_pred=classification, y_true=np.zeros_like(classification))
                    #cross_entropy = tf.reduce_sum((patches - classification)**2, axis=1)

                    # Distance from normal data
                    differences = patches[:, tf.newaxis, :] - batch_n[np.newaxis, : , :]
                    differences = differences ** 2
                    sample_distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(differences, axis=-1)), axis=1)
                    #sample_distance = tf.reduce_sum((patches - batch_n) ** 2, axis=1)
                    # Test with mahalanobis distance
                    #sample_distance = efficient_marahanobis(batch_n, patches, self.sigma)

                    # Sparsity                       axis=1)
                    sparsity = tf.math.sqrt(masks ** 2)
                    #sparsity = (patches - batch_a) ** 2
                    #sparsity = tf.sqrt(tf.reduce_sum(sparsity, axis=1))
                    #sparsity = tf.math.sqrt(np.abs(masks))
                    sparsity = tf.reduce_sum(sparsity, axis=1)

                    loss = tf.reduce_mean(loss_weights[0] * cross_entropy +
                                          loss_weights[1] * sample_distance +
                                          loss_weights[2] * sparsity +
                                          loss_weights[3] * ndim_loss +
                                          loss_weights[4] * dim_distance) #/ tf.reduce_sum(loss_weights)

                    tf.print(f'Loss: {loss}, Model:{tf.reduce_mean(cross_entropy)}, Dist: {tf.reduce_mean(sample_distance)}, '
                             f'Sparsity: {tf.reduce_mean(sparsity)}', f'Dim choise: {tf.reduce_mean(ndim_loss)}',
                             f'Subspace Distance: {tf.reduce_mean(dim_distance)}')

                model_vars = self.PATCH.trainable_variables
                #gradients = tape.gradient([loss, ndim_loss], model_vars)
                gradients = tape.gradient(loss, model_vars)
                self.optimizer.apply_gradients(zip(gradients, model_vars))

    def return_explanation(self, sample, threshold=0.7):
        sample = np.array(sample).reshape((1, -1))

        # compute model output
        mask, choose = self.MASKGEN(sample)
        choose = tf.where(choose > threshold, choose, 0)
        patches = self.MASKAPPLY([sample, mask, choose])

        return patches.numpy(), choose.numpy()
