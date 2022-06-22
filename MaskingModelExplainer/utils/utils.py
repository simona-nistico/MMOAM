import os
import pickle
from math import ceil

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.neighbors
import sklearn.metrics

from scipy import stats
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


def mal(x, mu, sigma):
    m_dist_x = np.dot((x - mu).transpose(), np.linalg.inv(sigma))
    m_dist_x = np.dot(m_dist_x, (x - mu))
    return 1 - stats.chi2.cdf(m_dist_x, 3)


def data_plot(data, labels, new_point=None, dimensions=None, name='gen_points', save=True, show=False, train=False,
              features_name=None):

    if dimensions is None:
        dimensions = np.arange(data.shape[1])

    if features_name is None:
        features_name = np.arange(data.shape[1])

    ndims = dimensions.shape[0]

    plt.figure(figsize=(4 * min(ndims, 2), 2.5 * ceil(ndims / 2)))
    for i in range(len(dimensions)):
        d = dimensions[i]
        plt.subplot(ceil(ndims / 2), min(ndims, 2), i+1)
        min_v = data[:, d].min()
        max_v = data[:, d].max()
        if train:
            min_v = min(min_v, new_point[d])
            max_v = max(max_v, new_point[d])
        offset = np.abs(max_v/10) if max_v != 0 else np.abs(min_v/10)
        plt.xlim(min_v - offset, max_v + offset)
        plt.scatter(data[labels == 0, d], np.full_like(data[labels == 0, d], fill_value=1.5), c='#C0Ca33', edgecolors='#AFB42B',
                    label='normal')
        plt.scatter(data[labels == 1, d], np.ones_like(data[labels == 1, d]), c='coral', edgecolors='orangered',
                    label='anomalous')
        if train:
            plt.scatter(new_point[d], np.full_like(new_point[d], fill_value=0.5), c='cornflowerblue', edgecolors='royalblue',
                        label='new point')
            yticksp_l = [0.5, 1., 1.5]
            yticksl_l = ['new point', 'anomalous', 'normal']
        else:
            yticksp_l = [1., 1.5]
            yticksl_l = ['anomalous', 'normal']
        plt.yticks(ticks=yticksp_l, labels=yticksl_l)
        plt.title(f'Dimension {features_name[d]}')
        plt.tight_layout()
        #plt.legend()
        i += 1
    if save:
        plt.savefig(name + '.eps')
        plt.savefig(name + '.jpg')
    if show:
        plt.show()
    else:
        plt.close()


def data_plot_clusters(data, labels, clusters, new_point=None, dimensions=None, name='gen_points', save=True, show=False, train=False,
              features_name=None):

    if dimensions is None:
        dimensions = np.arange(data.shape[1])

    if features_name is None:
        features_name = np.arange(data.shape[1])

    ndims = dimensions.shape[0]

    plt.figure(figsize=(4 * min(ndims, 5), 2.5 * ceil(ndims / 5)))
    for i in range(len(dimensions)):
        d = dimensions[i]
        plt.subplot(ceil(ndims / 5), min(ndims, 5), i+1)
        min_v = data[:, d].min()
        max_v = data[:, d].max()
        if train:
            min_v = min(min_v, new_point[d])
            max_v = max(max_v, new_point[d])
        offset = np.abs(max_v/10) if max_v != 0 else np.abs(min_v/10)
        plt.xlim(min_v - offset, max_v + offset)
        colormap = np.array(['r', 'g', 'b'])
        plt.scatter(data[labels == 0, d], np.full_like(data[labels == 0, d], fill_value=1.5), c=colormap[clusters[labels==0]], edgecolors='#AFB42B',
                    label='normal')
        plt.scatter(data[labels == 1, d], np.ones_like(data[labels == 1, d]), c='coral', edgecolors='orangered',
                    label='anomalous')
        if train:
            plt.scatter(new_point[d], np.full_like(new_point[d], fill_value=0.5), c='cornflowerblue', edgecolors='royalblue',
                        label='new point')
            yticksp_l = [0.5, 1., 1.5]
            yticksl_l = ['new point', 'anomalous', 'normal']
        else:
            yticksp_l = [1., 1.5]
            yticksl_l = ['anomalous', 'normal']
        plt.yticks(ticks=yticksp_l, labels=yticksl_l)
        plt.title(f'Dimension {features_name[d]}')
        plt.tight_layout()
        #plt.legend()
        i += 1
    if save:
        plt.savefig(name + '.eps')
        plt.savefig(name + '.jpg')
    if show:
        plt.show()
    else:
        plt.close()


def getclass(ti, tl, current_class, other_class):
    """
    Get a subset containing only images of the chosen couple of classes
    :param ti: images
    :param tl: labels
    :return: dataset: selected images
    :return: labels: labels of selected images
    :return: classes: chosen couple of classes
    :return: invclasses: dictionary for conversion to original class
    """
    classes = np.array([current_class, other_class])
    invclasses = dict()
    for i in range(len(classes)):
        invclasses[classes[i]] = i

    n = ti.shape[0]
    it = np.full(n, False)
    for c in classes:
        jt = tl == c
        it = it | jt
    tl = tl[it]
    for i in range(tl.shape[0]):
        tl[i] = invclasses[tl[i]]
    return ti[it], tl, classes, invclasses


def getneighds(img_id, train_images, train_labels, classes, ns=0, no=1, columns=None):
    """
    Get the image neighbourhood exploiting original dataset
    :param img_id: id of the image to expalin in the dataset
    :param train_images: image dataset
    :param train_labels: dataset labels
    :param classes: chosen couple of classes
    :param ns: number of sample of the same class
    :param no: number of sample of the other class
    :return:  train_images_expl: images for explanation
    :return:  train_labels_expl: labels for explanation
    """

    if columns is None:
        columns = np.arange(train_images.shape[1])

    cl = train_labels[img_id]
    print("Img Label:", classes[cl])
    cloth = (cl + 1) % 2
    print("Img Other Label:", classes[cloth])
    ido = train_labels == cloth
    ids = train_labels == cl
    ids[img_id] = False
    sh = train_images.shape
    shd = train_images[:, columns].shape
    nd = train_images.ndim
    #
    sh_expl = list(sh)
    sh_expl[0] = 1 + no + ns
    #
    train_images_expl = np.empty(sh_expl, dtype=np.object)
    train_labels_expl = np.zeros(sh_expl[0], dtype='int')
    #
    d = sk.metrics.euclidean_distances(
        train_images[img_id, columns].reshape(1, np.prod(shd[1:nd])),
        train_images[ido][:, columns].reshape(np.sum(ido), np.prod(shd[1:nd]))
    )
    sort = np.argsort(d)[0]
    img_expl_oth = sort[0:no]
    i = np.array([*range(sh[0])])
    i = i[ido]
    img_expl_oth = i[img_expl_oth]
    #
    if ns > 0:
        d = sk.metrics.euclidean_distances(
            train_images[img_id, columns].reshape(1, np.prod(shd[1:nd])),
            train_images[ids][:, columns].reshape(np.sum(ids), np.prod(shd[1:nd]))
        )
        sort = np.argsort(d)[0]
        img_expl_s = sort[0:ns]
        i = np.array([*range(sh[0])])
        i = i[ids]
        img_expl_s = i[img_expl_s]
        train_images_expl[1:1 + ns] = train_images[img_expl_s]
    #
    train_images_expl[0] = train_images[img_id]
    train_images_expl[1 + ns:1 + ns + no] = train_images[img_expl_oth]
    #
    # train_labels_expl[:] = cloth
    train_labels_expl[:ns + 1] = cl
    train_labels_expl[ns + 1:] = cloth
    return train_images_expl, train_labels_expl


def focal_loss(y_true, y_pred):
    gamma = 2.
    alpha = .11
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def _sod_2(self, X, pos):
    """This function is called internally to perform subspace outlier
    detection algorithm.

    Returns
    -------
    anomaly_scores : numpy array of shape (n_samples,)
        The anomaly score of the input samples.
    """
    ref_inds = self._snn(X)
    ref = X[ref_inds[pos,],]
    means = np.mean(ref, axis=0)  # mean of each column
    # average squared distance of the reference to the mean
    var_total = np.sum(np.sum(np.square(ref - means))) / self.ref_set_
    var_expect = self.alpha_ * var_total / X.shape[1]
    var_actual = np.var(ref, axis=0)  # variance of each attribute
    var_inds = [1 if (j < var_expect) else 0 for j in var_actual]

    return np.where(var_inds)[0]

def calcola_knn_score(ds, labels, pos):
    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(ds, labels)
    distances, indices = knn.kneighbors(ds)
    distances = distances.sum(axis=1)
    distances_idx = np.argsort(-1 * distances)
    return distances[pos], np.where(distances_idx==pos)[0][0], distances[distances_idx]


# def _sod_2(self, X, pos):
#     """This function is called internally to perform subspace outlier
#     detection algorithm.
#
#     Returns
#     -------
#     anomaly_scores : numpy array of shape (n_samples,)
#         The anomaly score of the input samples.
#     """
#     ref_inds = self._snn(X)
#     ref = X[ref_inds[pos,],]
#     means = np.mean(ref, axis=0)  # mean of each column
#     # average squared distance of the reference to the mean
#     var_total = np.sum(np.sum(np.square(ref - means))) / self.ref_set_
#     var_expect = self.alpha_ * var_total / X.shape[1]
#     var_actual = np.var(ref, axis=0)  # variance of each attribute
#     var_inds = [1 if (j < var_expect) else 0 for j in var_actual]

    return np.where(var_inds)[0]


def retrive_accuracies(data_path, results_path):
    precisions = []
    precisions_sod = []
    number = []
    number_sod = []

    for f in os.listdir(results_path):
        if f.endswith('_choose.joblib') and not f.endswith('_sod_choose.joblib'):
            print(f'FILE: {f}')
            data = pd.read_csv(data_path + f.replace('_choose.joblib', '.csv'))
            x_train = data.iloc[:, :data.shape[1] - 1].to_numpy()
            y_train = data['class']

            positions = []
            positions_sod = []
            number_f = []
            number_sod_f = []
            i = 0

            for ete in np.argwhere(data['class'].to_numpy() != 0):

                x_train_sub = x_train[y_train == 0]
                x_train_sub = np.append(x_train_sub, x_train[ete], axis=0)
                y_train_sub = y_train[y_train == 0]
                y_train_sub = np.append(y_train_sub, [1], axis=0)


                # SOD
                sod_choose = pickle.load(open(os.path.join(results_path, f.replace('_choose', '_sod_choose')), 'rb'))
                choose_s = sod_choose[i]
                number_sod_f.append(len(choose_s))

                # Our method
                f_choose = pickle.load(open(os.path.join(results_path, f), 'rb'))
                choose = f_choose[i]
                number_f.append(len(choose))

                # Compare results
                _, pos, _ = calcola_knn_score(x_train_sub[:, choose], y_train_sub, x_train_sub.shape[0]-1)
                _, pos_sod, _ = calcola_knn_score(x_train_sub[:, choose_s], y_train_sub, x_train_sub.shape[0] - 1)
                positions.append(pos)
                positions_sod.append(pos_sod)
                i += 1

            print(np.argwhere(np.array(positions) > 0)[0])
            precisions.append(np.where(np.array(positions)==0, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0)))
            precisions_sod.append(
                np.where(np.array(positions_sod) == 0, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0)))
            number.append(np.array(number_f).mean())
            number_sod.append(np.array(number_sod_f).mean())
            print(f"Ours: {np.where(np.array(positions)==0, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0))} ",
                  f"SOD: {np.where(np.array(positions_sod) == 0, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0))}")

    return np.array(precisions), np.array(precisions_sod), np.array(number), np.array(number_sod)


def compute_curve(point_pos, n_points):
    entries = np.zeros(n_points)
    entries[pd.Series(point_pos).value_counts().index.to_numpy()] = pd.Series(point_pos).value_counts()
    ecs = np.cumsum(entries) / len(point_pos)
    return ecs


def compute_positions_curve(data_path, results_path):
    ecs = []
    ecs_o = []
    ecs_sod = []
    ecs_np = []
    ecs_np_o = []
    ecs_np_sod = []
    length = []

    for f in os.listdir(results_path):
        if f.endswith('_choose.joblib') and not f.endswith('_sod_choose.joblib'):
            print(f'FILE: {f}')

            data = pd.read_csv(data_path + f.replace('_choose.joblib', '.csv'))
            x_train = data.iloc[:, :data.shape[1] - 1].to_numpy()
            y_train = data['class'].to_numpy()

            choose = pickle.load(open(os.path.join(results_path, f), 'rb'))
            sod_choose = pickle.load(open(os.path.join(results_path, f.replace('_choose', '_sod_choose')), 'rb'))
            new_points = pickle.load(open(os.path.join(results_path, f.replace('_choose', '_new_points')), 'rb'))

            points_pos = []
            points_pos_o = []
            points_pos_sod = []
            points_pos_np = []
            points_pos_np_o = []
            points_pos_np_sod = []
            length.append(x_train[y_train==0].shape[0]+1)
            i = 0

            for point in np.argwhere(y_train != 0):
                # Build dataset
                ds_point = x_train[y_train == 0]
                ds_point = np.append(ds_point, x_train[point], axis=0)
                ds_point_new = x_train[y_train == 0]
                ds_point_new = np.append(ds_point_new, new_points[i], axis=0)
                y_train_sub = np.zeros(ds_point.shape[0])
                y_train_sub[-1] = 1.

                # Compute scores
                _, pos, _ = calcola_knn_score(ds_point, y_train_sub, ds_point.shape[0] - 1)
                points_pos.append(pos)
                _, pos, _ = calcola_knn_score(ds_point[:, choose[i]], y_train_sub, ds_point.shape[0] - 1)
                points_pos_o.append(pos)
                _, pos, _ = calcola_knn_score(ds_point[:, sod_choose[i]], y_train_sub, ds_point.shape[0] - 1)
                points_pos_sod.append(pos)
                _, pos, _ = calcola_knn_score(ds_point_new, y_train_sub, ds_point.shape[0] - 1)
                points_pos_np.append(pos)
                _, pos, _ = calcola_knn_score(ds_point_new[:, choose[i]], y_train_sub, ds_point.shape[0] - 1)
                points_pos_np_o.append(pos)
                _, pos, _ = calcola_knn_score(ds_point_new[:, sod_choose[i]], y_train_sub, ds_point.shape[0] - 1)
                points_pos_np_sod.append(pos)

                i += 1

            ecs.append(compute_curve(points_pos, y_train[y_train==0].shape[0]+1))
            ecs_o.append(compute_curve(points_pos_o, y_train[y_train==0].shape[0]+1))
            ecs_sod.append(compute_curve(points_pos_sod, y_train[y_train==0].shape[0]+1))
            ecs_np.append(compute_curve(points_pos_np, y_train[y_train==0].shape[0]+1))
            ecs_np_o.append(compute_curve(points_pos_np_o, y_train[y_train==0].shape[0]+1))
            ecs_np_sod.append(compute_curve(points_pos_np_sod, y_train[y_train==0].shape[0]+1))

    return ecs, ecs_o, ecs_sod, ecs_np, ecs_np_o, ecs_np_sod, length


def compute_scores(curve):
    return curve.sum() / curve.shape


def kernel_fn(d, shape):
    kernel_width = np.sqrt(shape) * .75
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

def extract_points(x_train, y_train, current_class, other_class, ns, no):
    train_s, train_l, classes, invclasses = getclass(x_train, y_train.astype(np.int32), current_class,
                                                     other_class)

    img_id = np.where(train_l == 0)[0][0]

    test_images_expl, test_labels_expl = getneighds(img_id, train_s, train_l, classes, ns=ns, no=no)
    test_labels_expl = classes[test_labels_expl]
    test_images_expl = test_images_expl.astype(np.float32)

    return test_images_expl, test_labels_expl, train_s[img_id]


def compute_dists(reference_image, train_images, columns=None):
    """
    Get the image neighbourhood exploiting original dataset
    :param reference_image: id of the image to expalin in the dataset
    :param train_images: image dataset
    :param train_labels: dataset labels
    :return:  distances: images for explanation
    """

    if columns is None:
        columns = np.arange(train_images.shape[1])

    shd = train_images[:, columns].shape
    nd = train_images.ndim

    dists = sk.metrics.euclidean_distances(
        reference_image[columns].reshape(1, np.prod(shd[1:nd])),
        train_images[:, columns].reshape(shd[0], np.prod(shd[1:nd]))
    )

    return dists[0]


def generate_dataset(x_train, y_train, img_id, k=40, alpha=0.35):
    label = y_train[img_id]
    x_train_sub = x_train[(y_train == 0) | (y_train == label)]
    y_train_sub = y_train[(y_train == 0) | (y_train == label)]

    dists = compute_dists(x_train[img_id], x_train_sub[y_train_sub == 0])
    indexes = np.argsort(dists)

    #normal_points = np.empty((2 * k, x_train_sub.shape[1]))
    normal_points = x_train_sub[y_train_sub == 0][indexes[:2*k]]
    #normal_points[k:] = x_train_sub[y_train_sub == 0][(indexes[k:][np.random.randint(x_train_sub[y_train_sub == 0].shape[0] - k,
    #                                                               size=(k))])]

    lambda_v = alpha * (1 / np.sqrt(x_train.shape[1])) * dists[indexes][k]
    anomalous_points = np.random.normal(x_train[img_id], lambda_v, (2*k-1, x_train.shape[1]))
    anomalous_points = np.append(x_train[img_id:img_id+1], anomalous_points, axis=0)

    dataset = np.append(anomalous_points, normal_points, axis=0)
    labels = np.append(np.ones(anomalous_points.shape[0]), np.zeros(normal_points.shape[0]), axis=0)
    return dataset, labels

