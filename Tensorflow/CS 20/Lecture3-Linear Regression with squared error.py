import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

DATA_FILE = "data/birth_life_2010.txt"

# Step 1: read in data from the .txt file
data, n_samples = utils.read_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: construct model to predict Y (life expectancy from birth rate)
Y_predicted = w * X + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    # Step 8: train the model
    for i in range(10):
        for x, y in data:
            sess.run(optimizer, feed_dict={X:x, Y:y})

    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])
    print('w: %f, b: %f' %(w_out, b_out))

# plot the results
plt.plot(data[:,0], data[:,1], 'bo', label='Real Data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted Data ')
plt.legend()
plt.show()
