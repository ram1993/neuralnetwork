import tensorflow as tf
import numpy as np
tf.set_random_seed(234)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

x = tf.placeholder(tf.float32, shape = [None,2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([2,2]), name="weight1")
b1 = tf.Variable(tf.random_normal([2]), name="bias1")
h1 = tf.sigmoid(tf.matmul(x,w1) + b1)

w2 = tf.Variable(tf.random_normal([2,1]), name="weight2")
b2 = tf.Variable(tf.random_normal([1]), name="bias2")
h2 = tf.sigmoid(tf.matmul(h1,w2)+b2)

cost = -tf.reduce_mean(y*tf.log(h2) + (1-y)*tf.log(1-h2))

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

prediction = tf.cast(h2>0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={x:x_data, y:y_data})
        print(step, sess.run(cost, feed_dict={x:x_data, y:y_data}))
        #sess.run([w1,w2])

    hypothesis , p, a = sess.run([h2, prediction, accuracy], feed_dict={x:x_data, y:y_data})
    print(hypothesis)
    print(p)
    print(a)
