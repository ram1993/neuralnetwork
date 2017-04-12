import tensorflow as tf
import numpy as np
tf.set_random_seed(44)

dataset = np.loadtxt("zoo.csv", delimiter=',', dtype=np.float32)
x_data = dataset[:,:-1]
y_data = dataset[:,[-1]]

x = tf.placeholder(tf.float32, shape=[None,16])
yy = tf.placeholder(tf.int32, shape=[None, 1])

y = tf.one_hot(yy,7)
y = tf.reshape(y, [-1,7])

w = tf.Variable(tf.random_normal([16,7]), name="weight")
b = tf.Variable(tf.random_normal([7]), name="bias")

logits = tf.matmul(x,w)+b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

Prediction = tf.argmax(hypothesis, 1)

correct_prediction = tf.equal(Prediction, tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3000):
        _, cost_value, accuracy_value = sess.run([optimizer, cost, accuracy], feed_dict={x:x_data, yy:y_data})
        print(step, cost_value, accuracy_value)

    pred = sess.run(Prediction, feed_dict={x:x_data})

    for p,y in zip(pred, y_data.flatten()):
        print(p, y)
