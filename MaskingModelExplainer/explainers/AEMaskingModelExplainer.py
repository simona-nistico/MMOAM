from math import ceil

from tensorflow.python import keras
import tensorflow as tf
import numpy as np


class AEMaskingModelExplainer(tf.keras.Model):
    """
    Abstract class
    """

    def __init__(self, predict_fn, in_shape):
        """
        :param predict_fn: model to explain
        """
        super(AEMaskingModelExplainer, self).__init__()
        self.MASKGEN = self.defineMaskGen(in_shape)
        self.MASKAPPLY = self.defineMaskApply(in_shape)
        self.model = predict_fn

    def call(self, inputs, training=None, mask=None):
        img = self.img(inputs)
        mask = self.MASKGEN(img)
        patch = self.MASKAPPLY(mask)
        if training:
            model = self.model(patch)
            return [patch[0], patch[1], model]
        else:
            return patch

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True) # patch, choose, mask, model

            # Compute loss
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def defineMaskGen(self, in_shape):
        """
        Define the mask generator model, returns NotImplementedError if it is called from abstract class
        :param in_shape: input shape
        :return:
        """
        raise NotImplementedError('subclasses must override defineMaskGen!')

    def defineMaskApply(self, in_shape):
        """
        Define the mask applier model, returns NotImplementedError if it is called from abstract class
        :param in_shape: input shape
        :return:
        """

        raise NotImplementedError('subclasses must override defineMaskApply!')

    def test(self, id, classes, train_images, train_labels, drawplot=True):

        raise NotImplementedError('subclasses must override test!')

    def fit_explanator(self, train_images_expl, train_labels_expl, epochs=1, verbose=0,
                       sample_weight=None, batch_size=32, loss_weights=[1., 1.],
                       optimizer=tf.keras.optimizers.RMSprop()):
        raise NotImplementedError('subclasses must override fit_explanator!')

    def explain(self, train_images_expl, train_labels_expl, verbose=0, sample_weight=None, epochs=3000, batch_size=32,
                loss_weights=None, optimizer=tf.keras.optimizers.RMSprop()):
        """

        :param train_images_expl: Images for explanator training
        :param train_labels_expl: Labels for explanator training
        :param verbose: if True explainer training information will be show
        :param sample_weight: weight for each sample
        :param epochs: number of epochs
        :param loss_weights: weight for each loss
        :param optimizer: optimizer to use during training
        :return:
        """
        self.fit_explanator(train_images_expl, train_labels_expl, epochs=epochs, verbose=verbose, batch_size=batch_size,
                            sample_weight=sample_weight, loss_weights=loss_weights)
        print("done.")

