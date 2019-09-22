import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
train_data = tf.data.Dataset.from_tensor_slices(train)
test_data = tf.data.Dataset.from_tensor_slices(test)
train_data = train_data.batch(batch_size)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

# Step 3: create weights and bias
w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Step 4: build model
logits = tf.matmul(img, w) + b

# Step 5: define loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')

# Step 6: define training op
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    for i in range(n_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += 1
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))

writer.close()
