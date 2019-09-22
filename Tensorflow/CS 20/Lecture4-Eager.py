import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import matplotlib.pyplot as plt
import utils
import time

DATA_FILE = 'data/birth_life_2010.txt'

tfe.enable_eager_execution()

# Read the data into a dataset.
data, n_samples = utils.read_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

w = tfe.Variable(0.0)
b = tfe.Variable(0.0)

def prediction(x):
    return x * w + b

def train():
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    def loss_func(x, y):
        return utils.huber_loss(y, prediction(x), 10.0)

    grad_fn = tfe.implicit_value_and_gradients(loss_func)

    start = time.time()
    for epoch in range(50):
        total_loss = 0.0
        for x, y in tfe.Iterator(dataset):
            # y_pred = x * w + b
            loss, gradients = grad_fn(x, y)
            optimizer.apply_gradients(gradients)
            total_loss += loss
        if epoch % 10 == 0:
            print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))
    print('Took: %f seconds' % (time.time() - start))

train()
plt.plot(data[:,0], data[:,1], 'bo')
plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r', label="huber regression")
plt.legend()
plt.show()
