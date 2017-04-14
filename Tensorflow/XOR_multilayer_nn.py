import tensorflow as tf
import numpy as np
tf.set_random_seed(234)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

x = tf.placeholder(tf.float32, shape = [None,2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([2,10]), name="weight")
b1 = tf.Variable(tf.random_normal([10]), name="bias")
h1 = tf.sigmoid(tf.matmul(x,w1)+b1)



w2 = tf.Variable(tf.random_normal([10,10]), name="weight")
b2 = tf.Variable(tf.random_normal([10]), name="bias")
h2 = tf.sigmoid(tf.matmul(h1,w2)+b2)


w3 = tf.Variable(tf.random_normal([10,10]), name="weight")
b3 = tf.Variable(tf.random_normal([10]), name="bias")
h3 = tf.sigmoid(tf.matmul(h2,w3)+b3)


w4 = tf.Variable(tf.random_normal([10,1]), name="weight")
b4 = tf.Variable(tf.random_normal([1]), name="bias")
h4 = tf.sigmoid(tf.matmul(h3,w4)+b4)



cost = -tf.reduce_mean(y*tf.log(h4) + (1-y)*tf.log(1-h4))

train = tf.train.GradientDescentOptimizer(0.2).minimize(cost)

prediction = tf.cast(h4>0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, prediction), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        _, cost_value = sess.run([train,cost], feed_dict={x:x_data, y:y_data})
        #sess.run(w)
        print(step, cost_value)

    hypothesis , p, a = sess.run([h4, prediction, accuracy], feed_dict={x:x_data, y:y_data})
    print(hypothesis)
    print(p)
    print(a)
