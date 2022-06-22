import logging
import pickle
from math import ceil
from time import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from sklearn.datasets import make_blobs

from explainers import AETabularMM
from utils.utils import getclass, getneighds, focal_loss, data_plot, data_plot_clusters
from models.ad_models import define_ad_model_complex


def run_test(path, **kwargs):

    epochs = kwargs.pop('epochs')
    epochs_exp = kwargs.pop('exp_epochs')
    batch_exp = kwargs.pop('exp_batch')
    batch_size = kwargs.pop('batch_size')
    n_norm = kwargs.pop('n_samples_num')
    n_anorm = kwargs.pop('a_samples_num')
    n_adv = kwargs.pop('n_adv')
    n_dim = kwargs.pop('dim_number')
    # loss_weights = [1., 0.4, .5]
    loss_weights = kwargs.pop('loss_weights')
    n_std = kwargs.pop('n_std')
    centers = kwargs.pop('centers')
    dist_std = kwargs.pop('dist_std')
    ns = kwargs.pop('n_same')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    anom_dims = kwargs.pop('anom_dims')
    threshold = kwargs.pop('threshold')
    cl_number = kwargs.pop('cl_number')

    #path = f'logs/syntetic_{time_format(time())}'
    #os.mkdir(path)
    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), format='%(message)s', level=logging.INFO)

    # Generate clusters
    x_train, clusters = make_blobs(n_samples=n_norm+n_anorm,
                                    n_features=n_dim,
                                    centers=cl_number,
                                    cluster_std=n_std,
                                    shuffle=True)

    dims = np.random.permutation(np.arange(n_dim))
    # Create an anomalous sample
    logging.info('\nPoint before anormality introduction: \n' + str(x_train[-1]))
    #for d in anom_dims:
    mod_dir = np.random.randint(0, 2, (anom_dims)) * 2 - 1
    logging.info(f'Mod dir dim {dims[:anom_dims]}: {mod_dir}\n')
    x_train[n_norm:, dims[:anom_dims]] += mod_dir * dist_std * n_std
    x_train = x_train.astype(np.float32)

    logging.info('\nPoint after anormality introduction: \n' + str(x_train[-1]))

    y_train = np.zeros(x_train.shape[0], dtype=np.int32)
    y_train[n_norm:] = 1
    y_train = y_train.astype(np.float32)

    if n_dim == 2:
        colormap = np.array(['r', 'g', 'b'])
        plt.scatter(x_train[:x_train.shape[0]-1, 0], x_train[:x_train.shape[0]-1, 1], c=colormap[clusters[:x_train.shape[0]-1]])
        plt.scatter(x_train[x_train.shape[0] - 1, 0], x_train[x_train.shape[0] - 1, 1])
        plt.savefig(os.path.join(path, 'points.jpg'))
        plt.savefig(os.path.join(path, 'points.eps'))
        plt.show()
    else:
        data_plot_clusters(x_train, y_train, clusters, name=os.path.join(path, f'points'))

    in_shape = x_train.shape[1]

    # Define the ad model
    ad_model = define_ad_model_complex(x_train[0].shape)

    lr_schedule_ad = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=epochs*n_adv, decay_rate=0.3,
                                                                    staircase=True)
    lr_schedule_exp = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=epochs_exp, decay_rate=0.7,
                                                                     staircase=True)
    ad_model_opt = tf.keras.optimizers.Adam()
    exp_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule_exp)

    # first train
    ad_model.compile(optimizer=ad_model_opt, loss=focal_loss)

    # print(ad_model.evaluate(x_train, y_train))
    # print(np.where(ad_model.predict(x_train)[:, 0] < 0.5, 0, 1))

    results = []
    explanations = []
    masks = []

    # -------------------- Try with more samples ---------------------------
    current_class = 1
    other_class = 0
    train_s, train_l, classes, invclasses = getclass(x_train, y_train.astype(np.int32), current_class,
                                                     other_class)

    img_id = np.where(train_l == 0)[0][0]

    test_images_expl, test_labels_expl = getneighds(img_id, train_s, train_l, classes, ns=ns, no=no)
    test_labels_expl = classes[test_labels_expl]
    test_images_expl = test_images_expl.astype(np.float32)

    x_train_ext = test_images_expl.copy()
    y_train_ext = test_labels_expl.copy()

    explainer = AETabularMM.TabularMM(ad_model, in_shape, x_train_ext, optimizer=exp_opt)
    # ----------------------------------------------------------------------

    for i in range(n_adv):
        print('--------------------- ADV EPOCH: {} -------------------'.format(i))
        sample_to_explain = x_train[np.where(y_train == 1)]

        start_time = time()
        ad_model.trainable = True
        ad_model.fit(x_train_ext, y_train_ext, epochs=epochs, batch_size=batch_size, verbose=1)
        ad_model.trainable = False
        # Early-stopping
        pred = ad_model.predict(sample_to_explain)[:, 0]
        if pred < 0.5:
            logging.info(f'PRED: {pred}')
            logging.info(f'EARLY STOPPING EPOCH {i}')
            break

        exp_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule_exp)
        explainer.explain(test_images_expl, test_labels_expl, batch_size=batch_exp,
                          epochs=epochs_exp, loss_weights=loss_weights, optimizer=exp_opt)  # loss_weights=[1., 0.2, .4]
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
        x_train_ext = np.append(x_train_ext, new_sample, axis=0)
        y_train_ext = np.append(y_train_ext, [1.], axis=0)

        # ad_model.fit(x_train_ext, y_train_ext, epochs=epochs, batch_size=batch_size, verbose=0)

        data_plot(x_train, y_train, new_point=new_sample[0], name=os.path.join(path, f'adv_point_{i}'), train=True)
        data_plot(test_images_expl, test_labels_expl, new_point=new_sample[0], name=os.path.join(path, f'neighbourhood_{i}'), train=True)

        plt.figure(2, figsize=(4 * min(n_adv, 5), 3.8 * ceil(n_adv / 5)))
        plt.subplot(ceil(n_adv / 5), min(n_adv, 5), i + 1)
        plt.bar(np.arange(0, n_dim, dtype=np.int32), mask[0])
        plt.title('Try n.: {}'.format(i + 1))
        plt.tight_layout()
        plt.figure(3, figsize=(4 * min(n_adv, 5), 3.8 * ceil(n_adv / 5)))
        plt.subplot(ceil(n_adv / 5), min(n_adv, 5), i + 1)
        plt.bar(np.arange(0, n_dim, dtype=np.int32), choose[0])
        plt.title('Try n.: {}'.format(i + 1))
        plt.tight_layout()

        # print(np.where(ad_model.predict(x_train_ext)[:, 0] < 0.5, 0, 1))
        results.append(ad_model.predict(x_train_ext)[:, 0])
        print(ad_model.predict(x_train_ext)[:, 0])

    pickle.dump(x_train_ext, open(os.path.join(path, 'x_train_ext.joblib'), 'wb'))
    pickle.dump(y_train_ext, open(os.path.join(path, 'y_train_ext.joblib'), 'wb'))
    pickle.dump(test_images_expl, open(os.path.join(path, 'chosen_points_x.joblib'), 'wb'))
    pickle.dump(test_labels_expl, open(os.path.join(path, 'chosen_points_y.joblib'), 'wb'))
    pickle.dump(results, open(os.path.join(path, 'results.joblib'), 'wb'))
    pickle.dump(masks, open(os.path.join(path, 'masks.joblib'), 'wb'))
    plt.figure(2)
    plt.savefig(os.path.join(path, 'explanations.eps'))
    plt.savefig(os.path.join(path, 'explanations.jpg'))
    plt.figure(3)
    plt.savefig(os.path.join(path, 'choose.eps'))
    plt.savefig(os.path.join(path, 'choose.jpg'))
    #plt.show()
    new_sample, dims = explainer.return_explanation(sample_to_explain.reshape(1, -1), threshold=threshold)
    data_plot(x_train, y_train, new_point=new_sample[0], dimensions=np.where(dims == 1)[1],
              name=os.path.join(path, f'explanation'), train=True)
    plt.close('all')