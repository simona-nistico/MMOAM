import logging
import pickle
from time import time

import numpy as np
import tensorflow as tf
import os


from explainers import AETabularMM
from utils.utils import focal_loss, extract_points, _sod_2
from models.ad_models import define_ad_model
from pyod.models.sod import SOD

def run_test(path, **kwargs):

    epochs = kwargs.pop('epochs')
    epochs_exp = kwargs.pop('exp_epochs')
    batch_exp = kwargs.pop('exp_batch')
    batch_size = kwargs.pop('batch_size')
    n_norm = kwargs.pop('n_samples_num')
    n_anorm = kwargs.pop('a_samples_num')
    n_adv = kwargs.pop('n_adv')
    n_dim = kwargs.pop('dim_number')
    loss_weights = kwargs.pop('loss_weights')
    n_mean = kwargs.pop('n_mean')
    n_std = kwargs.pop('n_std')
    dist_std = kwargs.pop('dist_std')
    ns = kwargs.pop('n_same')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    anom_dims = kwargs.pop('anom_dims')
    threshold = kwargs.pop('threshold')
    n_runs = kwargs.pop('n_runs')

    chose_dims = []
    chose_dims_sod = []
    real_dims = []
    precisions = []
    precisions_sod = []
    recalls = []
    recalls_sod = []

    for i in range(n_runs):

        # Generate a syntetic dataset with mean 3 and variance 2
        x_train = np.random.normal(n_mean, n_std, (n_norm + n_anorm, n_dim))
        dims = np.random.permutation(np.arange(n_dim))
        # for d in anom_dims:
        mod_dir = np.random.randint(0, 2, (anom_dims)) * 2 - 1
        logging.info(f'Mod dir dim {dims[:anom_dims]}: {mod_dir}\n')
        x_train[n_norm:, dims[:anom_dims]] += mod_dir * dist_std * n_std
        real_dims.append(dims[:anom_dims])
        x_train = x_train.astype(np.float32)

        y_train = np.zeros(x_train.shape[0], dtype=np.int32)
        y_train[n_norm:] = 1
        y_train = y_train.astype(np.float32)

        in_shape = x_train.shape[1]

        # Define the ad model
        ad_model = define_ad_model(x_train[0].shape)
        ad_model_opt = tf.keras.optimizers.Adam()
        exp_opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # first train
        ad_model.compile(optimizer=ad_model_opt, loss=focal_loss)

        x_train_sub, y_train_sub, sample_to_explain = extract_points(x_train, y_train, other_class=0, current_class=1,
                                                                     no=no, ns=ns)

        x_train_ext = x_train_sub.copy()
        y_train_ext = y_train_sub.copy()

        explainer = AETabularMM.TabularMM(ad_model, in_shape, optimizer=exp_opt)

        for j in range(n_adv):
            print(f'--------------------- ADV EPOCH: {j} -------------------')
            sample_to_explain = x_train[np.where(y_train == 1)]
            start_time = time()
            ad_model.trainable = True
            ad_model.fit(x_train_ext, y_train_ext, batch_size=batch_size, epochs=epochs)
            ad_model.trainable = False
            # Early-stopping
            pred = ad_model.predict(sample_to_explain)[:, 0]
            if pred < 0.5:
                logging.info(f'PRED: {pred}')
                logging.info(f'EARLY STOPPING EPOCH {j}')
                break

            exp_opt = tf.keras.optimizers.Adam(learning_rate=lr)
            explainer.explain(x_train_sub, y_train_sub, batch_size=batch_exp,
                              epochs=epochs_exp, loss_weights=loss_weights,)
            new_sample = explainer.PATCH(sample_to_explain.reshape(1, -1))
            new_sample = new_sample.numpy()

            x_train_ext = np.append(x_train_ext, new_sample, axis=0)
            y_train_ext = np.append(y_train_ext, [1.], axis=0)

            tot_time = time() - start_time
            print('Elapsed time: ', tot_time)

        # Compute Explanations

        new_sample, dims_mm = explainer.return_explanation(sample_to_explain.reshape(1, -1), threshold=threshold)
        chose_dims.append(np.argwhere(dims_mm > threshold)[:, 1])
        print(list(chose_dims[i]))

        sod = SOD(n_neighbors=40, ref_set=20, alpha=0.9)
        sod.fit(x_train)
        dims_sod = _sod_2(sod, x_train, x_train.shape[0] - 1)
        chose_dims_sod.append(dims_sod)

        # Compute Scores

        anomalous_dims = set(dims[:anom_dims])
        dims_mm = set(list(chose_dims[i]))
        dims_sod = set(dims_sod)

        if len(dims_mm) == 0:
            precisions.append(-1)
            print('dims_mm empty')
        else:
            precisions.append(len(dims_mm.intersection(anomalous_dims))/len(dims_mm))
        if len(dims_sod) == 0:
            precisions_sod.append(-1)
            print('dims_sod empty')
        else:
            precisions_sod.append(len(dims_sod.intersection(anomalous_dims)) / len(dims_sod))
        recalls.append(len(dims_mm.intersection(anomalous_dims)) / len(anomalous_dims))
        recalls_sod.append(len(dims_sod.intersection(anomalous_dims)) / len(anomalous_dims))
        print(f'Precision mm: {precisions[i]} sod: {precisions_sod[i]}')
        print(f'Recall mm: {recalls[i]} sod: {recalls_sod[i]}')

    pickle.dump(chose_dims, open(os.path.join(path, 'chose_dims'), 'wb'))
    pickle.dump(chose_dims_sod, open(os.path.join(path, 'chose_dims_sod'), 'wb'))
    pickle.dump(real_dims, open(os.path.join(path, 'real_dims'), 'wb'))
    pickle.dump(precisions, open(os.path.join(path, 'precisions'), 'wb'))
    pickle.dump(precisions_sod, open(os.path.join(path, 'precisions_sod'), 'wb'))
    pickle.dump(recalls, open(os.path.join(path, 'recalls'), 'wb'))
    pickle.dump(recalls_sod, open(os.path.join(path, 'recalls_sod'), 'wb'))