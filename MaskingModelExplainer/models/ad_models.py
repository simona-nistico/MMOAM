from tensorflow import keras
import tensorflow as tf


def define_ad_model(in_shape):
    inputs = keras.layers.Input(in_shape)
    l = keras.layers.Dense(128, activation='relu')(inputs)
    l = keras.layers.Dropout(0.3)(l)
    l = keras.layers.Dense(64, activation='relu')(l)
    l = keras.layers.Dropout(0.3)(l)
    outputs = keras.layers.Dense(1, activation='sigmoid')(l)

    return keras.Model(inputs=inputs, outputs=outputs)

def define_ad_dummy(in_shape):
    inputs = keras.layers.Input(in_shape)
    l = keras.layers.Dense(32, activation='relu')(inputs)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dense(16, activation='relu')(l)
    l = keras.layers.BatchNormalization()(l)
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
    l = keras.layers.Dropout(0.4)(l)
    l = keras.layers.Dense(128, activation='relu')(l)
    l = keras.layers.Dropout(0.4)(l)
    l = keras.layers.Dense(256, activation='relu')(l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(0.3)(l)

    outputs = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(bias_init))(l)

    return keras.Model(inputs=inputs, outputs=outputs)
