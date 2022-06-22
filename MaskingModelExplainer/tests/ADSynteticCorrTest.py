import logging
import pickle
from math import ceil
from time import time
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from explainers import AETabularMM
from utils.utils import getclass, getneighds, focal_loss, data_plot
from models.ad_models import define_ad_model


def run_test(path, **kwargs):
    epochs = kwargs.pop('epochs')
    epochs_exp = kwargs.pop('exp_epochs')
    batch_size = kwargs.pop('batch_size')
    batch_exp = kwargs.pop('exp_batch')
    n_norm = kwargs.pop('n_samples_num')
    #n_anorm = kwargs.pop('a_samples_num')
    n_adv = kwargs.pop('n_adv')
    n_dim = kwargs.pop('dim_number')
    loss_weights = kwargs.pop('loss_weights')
    n_mean = kwargs.pop('n_mean')
    n_std = kwargs.pop('n_std')
    c_mean = kwargs.pop('c_mean')
    c_std = kwargs.pop('c_std')
    corr_deg = kwargs.pop('corr_deg')
    corr_anom = kwargs.pop('corr_anom')
    dist_std = kwargs.pop('dist_std')
    dist_std_2 = kwargs.pop('dist_std_2')
    ns = kwargs.pop('n_same')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    threshold = kwargs.pop('threshold')
    corr_dims = np.random.permutation(np.arange(n_dim))[:2]

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), level=logging.INFO, format='%(message)s')
    logging.info(f'Correlated dimentions: {corr_dims}\n')

    # Generate a syntetic dataset
    mean_vec = np.full(n_dim, fill_value=n_mean)
    mean_vec[corr_dims] = c_mean
    std_vec = np.full(n_dim, fill_value=n_std)
    std_vec[corr_dims] = c_std
    cov_mat = np.diag(std_vec)
    cov_mat[corr_dims[0], corr_dims[1]] = corr_deg
    cov_mat[corr_dims[1], corr_dims[0]] = corr_deg

    x_train = np.random.multivariate_normal(mean_vec, cov_mat, n_norm+1)
    x_train = x_train.astype(np.float32)

    # Create an anomalous sample
    logging.info('\nPoint before anormality introduction: \n' + str(x_train[-1]))
    mod_dir = np.random.randint(0, 2) * 2 - 1
    logging.info(f'Mod dir: {mod_dir}\n')
    if corr_anom == 'gen':
        x_train[n_norm:, corr_dims[0]] += mod_dir * dist_std * c_std
    elif corr_anom == 'corr':
        x_train[n_norm:, corr_dims[1]] += mod_dir * dist_std * c_std
    elif corr_anom == 'both':
        mod_dir2 = np.random.randint(0, 2) * 2 - 1
        logging.info(f'Mod dir second dim: {mod_dir2}\n')
        x_train[n_norm:, corr_dims[0]] += mod_dir * dist_std * c_std
        x_train[n_norm:, corr_dims[1]] += mod_dir2 * dist_std_2 * c_std
    else:
        raise NotImplementedError(f'Strategy {corr_anom} not implemented.')

    logging.info('\nPoint after anormality introduction: \n' + str(x_train[-1]))

    plt.scatter(x_train[:n_norm, corr_dims[0]], x_train[:n_norm, corr_dims[1]], label='normal')
    plt.scatter(x_train[n_norm:, corr_dims[0]], x_train[n_norm:, corr_dims[1]], label='anomal')
    plt.legend()
    plt.savefig(os.path.join(path, 'corr_dims.jpg'))
    plt.savefig(os.path.join(path, 'corr_dims.eps'))
    #plt.show()
    plt.close()

    y_train = np.zeros(x_train.shape[0], dtype=np.int32)
    y_train[n_norm:] = 1
    y_train = y_train.astype(np.float32)

    data_plot(x_train, y_train)

    in_shape = x_train.shape[1]

    # Define the ad model
    ad_model = define_ad_model(x_train[0].shape)

    lr_schedule_ad = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=epochs, decay_rate=0.96, staircase=True)
    lr_schedule_exp = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=epochs_exp, decay_rate=0.96, staircase=True)
    ad_model_opt = tf.keras.optimizers.Adam()
    exp_opt = tf.keras.optimizers.Adam(lr)

    # first train
    ad_model.compile(optimizer=ad_model_opt, loss=focal_loss)

    results = []
    explanations = []
    masks = []

    x_train_ext = x_train.copy()
    y_train_ext = y_train.copy()

    # -------------------- Try with more samples ---------------------------
    current_class = 1
    other_class = 0
    train_s, train_l, classes, invclasses = getclass(x_train, y_train.astype(np.int32), current_class,
                                                     other_class)

    img_id = np.where(train_l == 0)[0][0]

    test_images_expl, test_labels_expl = getneighds(img_id, train_s, train_l, classes, ns=ns, no=no)
    test_labels_expl = classes[test_labels_expl]
    test_images_expl = test_images_expl.astype(np.float32)

    explainer = AETabularMM.TabularMM(ad_model, in_shape, x_train, optimizer=exp_opt)
    # ----------------------------------------------------------------------

    for i in range(n_adv):
        print('--------------------- ADV EPOCH: {} -------------------'.format(i))
        sample_to_explain = x_train[np.where(y_train == 1)]
        
        start_time = time()
        ad_model.fit(x_train_ext, y_train_ext, epochs=epochs, batch_size=batch_size, verbose=0)
        # Early-stopping
        pred = ad_model.predict(sample_to_explain)[:, 0]
        if pred < 0.5:
            logging.info(f'PRED: {pred}')
            logging.info(f'EARLY STOPPING EPOCH {i}')
            break

        explainer.explain(test_images_expl, test_labels_expl, batch_size=batch_exp,
                          epochs=epochs_exp, loss_weights=loss_weights, optimizer=exp_opt) # loss_weights=[1., 0.2, .4]
        tot_time = time() - start_time
        print('Elapsed time: ', tot_time)
        logging.info(f'Elapsed time explanation {i}: {tot_time}')
        explanations.append(explainer)
        new_sample, _ = explainer.PATCH(sample_to_explain.reshape(1, -1))
        new_sample = new_sample.numpy()
        logging.info('Adversarial point {}: '.format(i) + str(new_sample))
        logging.info('Adversarial classification {}: '.format(i) + str(ad_model.predict(new_sample)))
        mask, choose = explainer.MASKGEN(sample_to_explain.reshape(1, -1))
        masks.append(mask.numpy())
        x_train_ext = np.append(x_train_ext, new_sample, axis=0)
        y_train_ext = np.append(y_train_ext, [1.], axis=0)

        #ad_model.fit(x_train_ext, y_train_ext, epochs=epochs, batch_size=batch_size, verbose=0)

        data_plot(x_train, y_train, new_point=new_sample[0], name=os.path.join(path, f'adv_point_{i}'), train=True)

        plt.figure(1, figsize=(4*min(n_adv, 5), 3.8*ceil(n_adv / 5)))
        plt.subplot(ceil(n_adv / 5), min(n_adv, 5), i + 1)
        plt.scatter(x_train[n_norm:, corr_dims[0]], x_train[n_norm:, corr_dims[1]], label='anomaly')
        plt.scatter(x_train[:n_norm, corr_dims[0]], x_train[:n_norm, corr_dims[1]], label='normal')
        plt.scatter(new_sample[:, corr_dims[0]], new_sample[:, corr_dims[1]], label='new point')
        plt.tight_layout()
        plt.figure(2, figsize=(4*min(n_adv, 5), 3.8*ceil(n_adv / 5)))
        plt.subplot(ceil(n_adv/ 5), min(n_adv, 5), i+1)
        plt.bar(np.arange(0, n_dim, dtype=np.int32), mask[0])
        plt.title('Try n.: {}'.format(i+1))
        plt.tight_layout()
        plt.figure(3, figsize=(4 * min(n_adv, 5), 3.8 * ceil(n_adv / 5)))
        plt.subplot(ceil(n_adv / 5), min(n_adv, 5), i + 1)
        plt.bar(np.arange(0, n_dim, dtype=np.int32), choose[0])
        plt.title('Try n.: {}'.format(i + 1))
        plt.tight_layout()

        #print(np.where(ad_model.predict(x_train_ext)[:, 0] < 0.5, 0, 1))
        results.append(np.where(ad_model.predict(x_train_ext)[:, 0] < 0.5, 0, 1))

    # Take results
    new_sample, _ = explainer.PATCH(sample_to_explain.reshape(1, -1))
    new_sample = new_sample.numpy()

    pickle.dump(x_train_ext, open(os.path.join(path, 'x_train_ext.joblib'), 'wb'))
    pickle.dump(y_train_ext, open(os.path.join(path, 'y_train_ext.joblib'), 'wb'))
    pickle.dump(results, open(os.path.join(path, 'results.joblib'), 'wb'))
    pickle.dump(masks, open(os.path.join(path, 'masks.joblib'), 'wb'))
    plt.figure(1)
    #plt.legend()
    plt.savefig(os.path.join(path, 'adv_samples.eps'))
    plt.savefig(os.path.join(path, 'adv_samples.jpg'))
    plt.figure(2)
    plt.savefig(os.path.join(path, 'explanations.eps'))
    plt.savefig(os.path.join(path, 'explanations.jpg'))
    plt.figure(3)
    plt.savefig(os.path.join(path, 'choose.eps'))
    plt.savefig(os.path.join(path, 'choose.jpg'))
    plt.figure(4)
    plt.scatter(x_train[n_norm:, corr_dims[0]], x_train[n_norm:, corr_dims[1]], label='anomalous',  c='coral', edgecolors='orangered')
    plt.scatter(x_train[:n_norm, corr_dims[0]], x_train[:n_norm, corr_dims[1]], label='normal', c='palegreen', edgecolors='lightgreen')
    plt.scatter(new_sample[:, corr_dims[0]], new_sample[:, corr_dims[1]], label='new point', c='cornflowerblue', edgecolors='royalblue')
    plt.savefig(os.path.join(path, 'adv_sample.eps'))
    plt.savefig(os.path.join(path, 'adv_sample.jpg'))
    #plt.show()
    new_sample, dims = explainer.return_explanation(sample_to_explain.reshape(1, -1), threshold=threshold)
    data_plot(x_train, y_train, new_point=new_sample[0], dimensions=np.where(dims==1)[1],
              name=os.path.join(path, f'explanation'), train=True)
    plt.close('all')
