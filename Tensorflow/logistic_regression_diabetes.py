# logistic regression example with diabetes examples
import tensorflow as tf
import numpy as np
tf.set_random_seed(4)

dataset = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = dataset[:,:-1]
y_data = dataset[:,[-1]]

print x_data.shape
print y_data.shape

x = tf.placeholder(tf.float32, shape=[None,8])
y = tf.placeholder(tf.float32, shape= [None, 1])

w = tf.Variable(tf.random_normal([8,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

h = tf.sigmoid(tf.matmul(x,w)+b)

cost = -tf.reduce_mean( y*tf.log(h) + (1-y)*tf.log(1-h))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

Prediction = tf.cast(h>0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(Prediction, y),dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):

        _, cost_value, accuracy_value, Prediction_value, hypothesis_value=\
        sess.run([train, cost, accuracy, Prediction, h], feed_dict={x:x_data, y:y_data})
        print(step, cost_value, accuracy_value)
    print(hypothesis_value, Prediction_value)
