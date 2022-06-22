import os
from time import time

import tensorflow as tf
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Dense, Flatten

from utils.utils import getclass, getneighds

tf.config.experimental_run_functions_eagerly(True)
import numpy as np
import explainers.ImageMMSegments as mmi


def build_model(in_shape):
    model = Sequential()
    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model_path = 'models/simple_model.hdf5'

    ds = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = ds.load_data()
    in_shape = train_images[0].shape
    train_images = train_images / 255
    test_images = test_images / 255

    #bbox = SVC(probability=True)
    #print('---------- START TRAINING -----------')
    #bbox.fit(train_images.reshape(train_images.shape[0],-1), train_labels)
    #bbox = pickle.load(open('bbox_svc.joblib', 'rb'))
    #print('---------- STOP TRAINING -----------')
    #black_box = lambda x: bbox.predict_proba(tf.reshape(x, (tf.shape(x)[0], -1)))

    if os.path.exists(model_path):
        print('Loading model...')
        black_box = build_model(in_shape)
        black_box.load_weights(model_path)
    else:
        print('Training model..')
        black_box = build_model(in_shape)
        black_box.fit(train_images, train_labels, epochs=10, batch_size=256)
        black_box.save_weights(model_path)
    print('Done!')

    black_box.trainable = False
    # black_box.evaluate(test_images, test_labels)

    current_class = 8
    other_class = 9
    epochs = 400

    # Explain
    explainer = mmi.MMImageSegments(black_box, in_shape)

    # ------------------------------- Prepare data -----------------------------------
    test_images, test_labels, classes, invclasses = getclass(test_images, test_labels, current_class, other_class)

    img_id = np.where(test_labels == 0)[0][0]
    ns = 0
    no = 0

    test_images_expl, test_labels_expl = getneighds(img_id, test_images, test_labels, classes, ns=ns, no=no)
    test_labels_expl = classes[test_labels_expl]

    # ---------------------------------- Explain -----------------------------------------
    start_time = time()
    explainer.explain(test_images_expl, np.full(test_images_expl.shape[0], fill_value=current_class), epochs=epochs,
                      loss_weights=[1., 0.0, 0.5])
    print('Elapsed time: ', time() - start_time)

    # ------------------------------------ Test -------------------------------------------
    mask, patch = explainer.test(img_id, classes, test_images, test_labels, True)
    print('Model prediction: ', black_box.predict(np.array([patch])))