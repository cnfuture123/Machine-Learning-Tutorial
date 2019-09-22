import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import time

DATA_FILE = "data/birth_life_2010.txt"

# Step 1: read in data from the .txt file
data, n_samples = utils.read_data(DATA_FILE)

# Step 2: create tf.data.Dataset object and iterate dataset using iterator
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

# Step 3: create weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: construct model to predict Y (life expectancy from birth rate)
Y_predicted = w * X + b

# Step 5: use the huber error as the loss function
loss = utils.huber_loss(Y, Y_predicted, 10.0)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

start = time.time()
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # Step 8: train the model
    for i in range(50):
        sess.run(iterator.initializer)
        total_loss = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += 1
        except tf.errors.OutOfRangeError:
            pass

        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])
    print('w: %f, b: %f' %(w_out, b_out))

print('Took: %f seconds' %(time.time() - start))

# plot the results
plt.plot(data[:,0], data[:,1], 'bo', label='Real Data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted Data with huber loss')
plt.legend()
plt.show()
