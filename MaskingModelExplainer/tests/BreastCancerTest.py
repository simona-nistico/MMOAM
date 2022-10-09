import logging
import pickle
from time import time
import os

import numpy as np
import sklearn
import tensorflow as tf
import pandas as pd
from pyod.models.sod import SOD

from explainers import AETabularMM
from utils.utils import getclass, getneighds, focal_loss, _sod_2, calcola_knn_score
from models.ad_models import define_ad_model_complex


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

    data = pd.read_csv('datasets/breast_cancer/data.csv')
    scaler = sklearn.preprocessing.StandardScaler()
    x_train = data.iloc[:, 2:data.shape[1] - 1].to_numpy()
    x_train = scaler.fit_transform(x_train)
    y_train = np.where(data.iloc[:, 1] == 'M', 1., 0.)
    current_class = 1
    other_class = 0

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), level=logging.INFO, format='%(message)s')

    distances = []
    distances_sod = []

    results = []
    f_choose = []
    sod_choose = []

    positions = []
    positions_sod = []
    new_points = []

    for ete in np.argwhere(y_train == 1.0):
        ad_model_opt = tf.keras.optimizers.Adam()
        exp_opt = tf.keras.optimizers.Adam(learning_rate=lr)

        x_train_sub = x_train[y_train == 0]
        x_train_sub = np.append(x_train_sub, x_train[ete], axis=0)
        y_train_sub = y_train[y_train == 0]
        y_train_sub = np.append(y_train_sub, [1.], axis=0)

        # -------------------- Try with more samples ---------------------------
        train_s, train_l, classes, invclasses = getclass(x_train_sub, y_train_sub.astype(np.int32), current_class,
                                                         other_class)

        img_id = np.where(train_l == 0)[0][0]

        test_images_expl, test_labels_expl = getneighds(img_id, train_s, train_l, classes, ns=ns, no=no)
        test_labels_expl = classes[test_labels_expl]
        test_labels_expl = np.where(test_labels_expl == 0, 0., 1.)
        test_images_expl = test_images_expl.astype(np.float32)

        x_train_sel = test_images_expl.copy()
        y_train_sel = test_labels_expl.copy()

        in_shape = x_train.shape[1]

        # Define the ad model
        ad_model = define_ad_model_complex(in_shape)
        ad_model.compile(optimizer=ad_model_opt, loss=focal_loss, metrics=['accuracy'])

        explainer = AETabularMM.TabularMM(ad_model, in_shape, test_images_expl, optimizer=exp_opt,
                                          weights=None)

        logging.info(f'EXPLAINING POINT: {ete}')
        print(f'EXPLAINING POINT: {ete}')

        sample_to_explain = train_s[img_id: img_id + 1]
        #
        for i in range(n_adv):
            print('--------------------- ADV EPOCH: {} -------------------'.format(i))
            start_time = time()
            explainer.model.trainable = True
            explainer.model.fit(x_train_sel, y_train_sel, epochs=epochs, batch_size=batch_size, verbose=0)
            explainer.model.trainable = False
            print(explainer.model.evaluate(x_train_sel, y_train_sel))

            # Early-stopping
            pred = explainer.model.predict(sample_to_explain)
            if pred < 0.5:
                logging.info(f'PRED: {pred}')
                logging.info(f'EARLY STOPPING EPOCH {i}')
                print(f'EARLY STOPPING EPOCH {i}')
                break

            exp_opt = tf.keras.optimizers.Adam(learning_rate=lr)
            explainer.explain(test_images_expl, test_labels_expl, batch_size=batch_exp,
                              epochs=epochs_exp, loss_weights=loss_weights,
                              optimizer=exp_opt)
            tot_time = time() - start_time

            print('Elapsed time: ', tot_time)
            logging.info(f'Elapsed time explanation {i}: {tot_time}')

            new_sample = explainer.PATCH(sample_to_explain.reshape(1, -1))
            logging.info('Adversarial point {}: '.format(i) + str(new_sample))
            logging.info('Adversarial classification {}: '.format(i) + str(ad_model.predict(new_sample)))
            print(explainer.model.predict(new_sample))
            x_train_sel = np.append(x_train_sel, new_sample, axis=0)
            y_train_sel = np.append(y_train_sel, [1.], axis=0)

            results.append(np.where(ad_model.predict(new_sample)[:, 0] < 0.5, 0, 1))

        # Use SOD
        sod = SOD(n_neighbors=40, ref_set=20, alpha=0.9)
        sod.fit(x_train_sub)
        choose_s = _sod_2(sod, x_train_sub, x_train_sub.shape[0] - 1)
        sod_choose.append(choose_s)

        # Our method
        new_sample, choose = explainer.return_explanation(sample_to_explain.reshape(1, -1), threshold)
        choose = np.argwhere(choose >= threshold)[:, 1].reshape(-1)
        f_choose.append(choose)
        new_points.append(new_sample)

        # Compare results
        dists, pos, _ = calcola_knn_score(x_train_sub[:, choose], y_train_sub, x_train_sub.shape[0]-1)
        dists_sod, pos_sod, _ = calcola_knn_score(x_train_sub[:, choose_s], y_train_sub, x_train_sub.shape[0] - 1)
        positions.append(pos)
        positions_sod.append(pos_sod)
        distances.append(dists)
        distances_sod.append(dists_sod)

    pickle.dump(f_choose, open(os.path.join(path, 'choose.joblib'), 'wb'))
    pickle.dump(sod_choose, open(os.path.join(path,'sod_choose.joblib'), 'wb'))
    pickle.dump(new_points, open(os.path.join(path, 'new_points.joblib'), 'wb'))

    pickle.dump(positions_sod, open(os.path.join(path, 'test_positions_sod.joblib'), 'wb'))
    pickle.dump(positions, open(os.path.join(path, 'test_positions.joblib'), 'wb'))
    pickle.dump(distances_sod, open(os.path.join(path, 'test_distances_sod.joblib'), 'wb'))
    pickle.dump(distances, open(os.path.join(path, 'test_distances.joblib'), 'wb'))
    print('TOP 1')
    print(
        f"Ours: {np.where(np.array(positions) == 0, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0))} ",
        f"SOD: {np.where(np.array(positions_sod) == 0, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0))}"
    )
    print('TOP 5')
    print(
        f"Ours: {np.where(np.array(positions) < 5, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0))} ",
        f"SOD: {np.where(np.array(positions_sod) < 5, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0))}"
    )
