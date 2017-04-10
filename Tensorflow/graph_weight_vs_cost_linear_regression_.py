#graph between weight and cost for linear regrassion
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(2)

x = [1,2,3,4,5]
y = [2,4,6,8,10]

w = tf.placeholder(tf.float32)

h = w*x

cost = tf.reduce_mean(tf.square(h-y))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

cost_values = []
weight_values = []

for w_i in  range(-10,50,1):
    weight_i = w_i*0.1
    cost_v, weight_v = sess.run([cost, w], feed_dict={w: weight_i})
    cost_values.append(cost_v)
    weight_values.append(weight_v)
    print(weight_v, cost_v, w_i)

plt.plot(weight_values, cost_values)
plt.show()
