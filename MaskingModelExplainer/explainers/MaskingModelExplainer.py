from math import ceil

from tensorflow import keras
import tensorflow as tf
import numpy as np


class MaskingModelExplainer():
    """
    Abstract class
    """

    def __init__(self, predict_fn):
        """
        :param predict_fn: model to explain
        """
        self.MASKGEN = None
        self.MASKAPPLY = None
        self.PATCH = None
        self.model = predict_fn

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

    def definePatch(self, in_shape):
        """
        Define the model that produce the patch from the original image
        :param in_shape: input shape
        :return:
        """

        # Mette insieme maskapply e maskgen
        img = keras.Input(shape=in_shape)
        limeMASKGEN = self.defineMaskGen(in_shape)
        limeMASKAPPLY = self.defineMaskApply(in_shape)
        mask = limeMASKGEN(img)
        patch = limeMASKAPPLY([img, mask])
        # Generatore patch - Applicatore della patch - Modello completo (gen+app)
        return limeMASKGEN, limeMASKAPPLY, keras.Model(inputs=img, outputs=patch)

    def test(self, id, classes, train_images, train_labels, drawplot=True):

        raise NotImplementedError('subclasses must override test!')

    def buildExplanator(self, predict_fn, in_shape):
        """
        Build the explanator model
        :param predict_fn: model to explain
        :param in_shape: input shape
        :return:
        """
        img = keras.Input(shape=in_shape)
        limeMASKGEN, limeMASKAPPLY, limePATCH = self.definePatch(in_shape)
        #patch = limePATCH(img)
        # outputs = predict_fn(patch)
        # Output di prima + modello che prende in input l'immagine, genera ed applica la patch e poi dà in output
        # la classificazione del modello
        return limeMASKGEN, limeMASKAPPLY, limePATCH

    def fit_explanator(self, train_images_expl, train_labels_expl, epochs=1, verbose=False, batch_size=32,
                       loss_weights=None):
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
                            sample_weight=sample_weight, loss_weights=loss_weights, optimizer=optimizer)
        print("done.")
