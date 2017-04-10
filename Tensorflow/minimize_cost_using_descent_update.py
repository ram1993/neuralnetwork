import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(5)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]), name="weight")

h = w*x

learning_rate = 0.01
cost = tf.reduce_mean(tf.square(h-y))
gradient = tf.reduce_mean((w*x-y)*x)
decent = w - learning_rate*gradient
update = w.assign(decent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_data = [1,2,3,4,5,6]
y_data = [2,4,6,8,10,12]

cost_list = []
weight_list = []

for step in range(100):

    cost_value,weight_value, delta, _  = sess.run([cost, w, gradient, update], feed_dict={x:x_data, y: y_data})
    cost_list.append(cost_value)
    weight_list.append(weight_value)
    #sess.run(cost, feed_dict={x:x_data, y: y_data})
    #sess.run(update, feed_dict={x:x_data, y: y_data})
    print("step:{0}, change: {1}, weight: {2}, cost: {3}".format(step, delta*learning_rate, weight_value, cost_value))

plt.plot(weight_list, cost_list)
plt.show()
