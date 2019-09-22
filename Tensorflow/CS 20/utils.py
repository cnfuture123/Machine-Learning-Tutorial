import os
import gzip
import shutil
import struct
import urllib

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def read_data(filename):
    """
    :param filename
    :return: array
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    life = [float(line[2]) for line in data]
    data = list(zip(births, life))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples

def huber_loss(y, y_pred, delta):
    diff = tf.abs(y - y_pred)
    def f1(): return 0.5 * tf.square(diff)
    def f2(): return delta * diff - 0.5 * tf.square(delta)
    return tf.cond(diff < delta, f1, f2)

def download_mnist(path):
    safe_mkdir(path)
    url = 'http://yann.lecun.com/exdb/mnist'
    filenames = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
    expected_bytes = [9912422, 28881, 1648877, 4542]
    for filename, byte in zip(filenames, expected_bytes):
        download_url = os.path.join(url, filename)
        download_url = download_url.replace('\\', '/')
        # download_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        local_path = os.path.join(path, filename)
        download_file(download_url, local_path, byte, True)

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def download_file(download_url, local_path, expected_byte=None, unzip_and_remove=False):
    """
    Download the file from download_url into local_dest
    if the file doesn't already exists.
    If expected_byte is provided, check if
    the downloaded file has the same number of bytes.
    If unzip_and_remove is True, unzip the file and remove the zip file
    """
    if os.path.exists(local_path) or os.path.exists(local_path[:-3]):
        print('%s already exists' %local_path)
    else:
        print('Downloading %s' %download_url)
        local_file, _ = urllib.request.urlretrieve(download_url, local_path)
        file_stat = os.stat(local_path)
        if expected_byte:
            if file_stat.st_size == expected_byte:
                print('Successfully downloaded %s' %local_path)
                if unzip_and_remove:
                    with gzip.open(local_path, 'rb') as f_in, open(local_path[:-3],'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(local_path)
            else:
                print('The downloaded file has unexpected number of bytes')

def read_mnist(path, flatten=True, num_train=55000):
    imgs, labels = parse_data(path, 'train', flatten)
    indices = np.random.permutation(labels.shape[0])
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    train_img, train_labels = imgs[train_idx, :], labels[train_idx, :]
    val_img, val_labels = imgs[val_idx, :], labels[val_idx, :]
    test = parse_data(path, 't10k', flatten)
    return (train_img, train_labels), (val_img, val_labels), test

def parse_data(path, dataset, flatten):
    if dataset != 'train' and dataset != 't10k':
        raise NameError('dataset must be train or t10k')

    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8) #int8
        new_labels = np.zeros((num, 10))
        new_labels[np.arange(num), labels] = 1

    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols) #uint8
        imgs = imgs.astype(np.float32) / 255.0
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, new_labels

