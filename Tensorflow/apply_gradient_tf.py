# play with gradient in tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1,2,3,4,5,6]
y_train = [4,8,12,16,20,24]

w = tf.Variable(10000.0)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

h = w*x
gradient = tf.reduce_mean((w*x-y)**2)
cost = tf.reduce_mean(tf.square(h-y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

gvs = optimizer.compute_gradients(cost, [w])

apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(100):
    print(step,sess.run([cost, gradient, w ,gvs], feed_dict= {x:x_train, y: y_train} ) )
    sess.run(apply_gradients, feed_dict= {x:x_train, y: y_train})
