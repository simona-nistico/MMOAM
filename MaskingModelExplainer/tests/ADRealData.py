import logging
import pickle
from time import time
import os
import scipy.io

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from explainers import AETabularMM
from utils.utils import getclass, getneighds, focal_loss, data_plot
from models.ad_models import define_ad_model_complex  # define_ad_model_complex


def run_test(path, **kwargs):
    epochs = kwargs.pop('epochs')
    epochs_exp = kwargs.pop('exp_epochs')
    batch_size = kwargs.pop('batch_size')
    batch_exp = kwargs.pop('exp_batch')
    n_adv = kwargs.pop('n_adv')
    loss_weights = kwargs.pop('loss_weights')
    ns = kwargs.pop('n_same')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    threshold = kwargs.pop('threshold')
    dataset = kwargs.pop('dataset')
    n_class = kwargs.pop('n_class')

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), level=logging.INFO, format='%(message)s')

    if dataset == 'wbc':
        data = scipy.io.loadmat('datasets/ad/wbc.mat')
        columns = None
        x_train = data['X']
        y_train = data['y'].reshape(-1)
        current_class = 1
        other_class = 0
        c_columns = None
    elif dataset == 'ionosphere':
        # TODO
        return
    elif dataset == 'iris':
        data = pd.read_csv('datasets/ad/iris.csv')
        columns = data.columns
        x_train = data.iloc[:, :4].to_numpy()
        y_train = data['Species']
        current_class = 1
        other_class = 0
        c_columns = None
    elif dataset == 'wine':
        data = pd.read_csv('datasets/ad/winequality-red.csv')
        columns = data.columns
        x_train = data.iloc[:, :11].to_numpy()
        y_train = data['quality']
        current_class = 1
        other_class = 0
        c_columns = None
    elif dataset == 'zoo':
        data = pd.read_csv('datasets/zoo/zoo_mod.csv')
        columns = data.columns[1:]
        x_train = data[(data['class_type']==0) | (data['class_type']==n_class)].iloc[:, :17]
        x_train = x_train.to_numpy()
        y_train = data[(data['class_type']==0) | (data['class_type']==n_class)]['class_type']
        y_train = np.where(y_train == 0, 1., 0.)
        current_class = 1
        other_class = 0
        c_columns = np.arange(1, x_train.shape[1])
    else:
        raise NotImplementedError(f'Dataset {dataset} not yet implemented')

    ad_model_opt = tf.keras.optimizers.Adam()
    exp_opt = tf.keras.optimizers.Adam(learning_rate=lr)

    explanations = []
    masks = []

    # -------------------- Try with more samples ---------------------------
    train_s, train_l, classes, invclasses = getclass(x_train, y_train.astype(np.int32), current_class,
                                                     other_class)

    img_id = np.where(train_l == 0)[0][0]

    test_images_expl, test_labels_expl = getneighds(img_id, train_s, train_l, classes, ns=ns, no=no, columns=c_columns)
    test_labels_expl = classes[test_labels_expl]

    if dataset == 'zoo':
        train_s = train_s[:, 1:]
        train_s = train_s.astype(np.float32)
        pickle.dump(test_images_expl[:,0], open('neighbours.pickle','wb'))
        test_images_expl = test_images_expl[:,1:]
        x_train = x_train[:, 1:]

    test_images_expl = test_images_expl.astype(np.float32)

    x_train = x_train.astype(np.float32)
    sample_to_explain = train_s[img_id: img_id + 1]
    # ----------------------------------------------------------------------

    x_train_ext = test_images_expl.copy()
    y_train_ext = test_labels_expl.copy()

    in_shape = x_train.shape[1]

    data_plot(x_train, y_train, features_name=columns)

    # Define the ad model
    ad_model = define_ad_model_complex(x_train[0].shape)
    ad_model.compile(optimizer=ad_model_opt, loss=focal_loss)

    explainer = AETabularMM.TabularMM(ad_model, in_shape, optimizer=exp_opt)

    for i in range(n_adv):
        print('--------------------- ADV EPOCH: {} -------------------'.format(i))

        start_time = time()
        ad_model.trainable = True
        ad_model.fit(x_train_ext, x_train_ext, epochs=epochs, batch_size=batch_size, verbose=1)
        ad_model.trainable = False
        ad_model.evaluate(x_train_ext, x_train_ext)

        Early-stopping
        pred = ad_model.predict(sample_to_explain)[:, 0]
        if pred < 0.5:
            logging.info(f'PRED: {pred}')
            logging.info(f'EARLY STOPPING EPOCH {i}')
            break

        explainer.explain(test_images_expl, test_labels_expl, batch_size=batch_exp,
                          epochs=epochs_exp, loss_weights=loss_weights)  # loss_weights=[1., 0.2, .4]
        tot_time = time() - start_time

        print('Elapsed time: ', tot_time)
        logging.info(f'Elapsed time explanation {i}: {tot_time}')

        explanations.append(explainer)
        new_sample = explainer.PATCH(sample_to_explain.reshape(1, -1))
        new_sample = new_sample.numpy()
        logging.info('Adversarial point {}: '.format(i) + str(new_sample))
        logging.info('Adversarial classification {}: '.format(i) + str(ad_model.predict(new_sample)))
        mask, choose = explainer.MASKGEN(sample_to_explain.reshape(1, -1))
        masks.append(mask.numpy())
        print('MASK: ', mask.numpy())
        print('CHOOSE: ', choose.numpy())
        x_train_ext = np.append(x_train_ext, new_sample, axis=0)
        y_train_ext = np.append(y_train_ext, [1.], axis=0)

        data_plot(x_train, y_train, new_point=new_sample[0], name=os.path.join(path, f'adv_point_{i}'), train=True, features_name=columns)
        data_plot(test_images_expl, test_labels_expl, new_point=new_sample[0], name=os.path.join(path, f'adv_point_subset{i}'), train=True, features_name=columns)


    pickle.dump(x_train_ext, open(os.path.join(path, 'x_train_ext.joblib'), 'wb'))
    pickle.dump(y_train_ext, open(os.path.join(path, 'y_train_ext.joblib'), 'wb'))
    pickle.dump(masks, open(os.path.join(path, 'masks.joblib'), 'wb'))

    new_sample, dims = explainer.return_explanation(sample_to_explain.reshape(1, -1), threshold=threshold)
    print(dims)
    data_plot(test_images_expl, test_labels_expl, new_point=new_sample[0], dimensions=np.where(dims > threshold)[1],
              name=os.path.join(path, f'explanation'), train=True, features_name=columns)
    plt.close('all')