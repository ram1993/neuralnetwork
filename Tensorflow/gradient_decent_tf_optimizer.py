# we can use in bulit tensorflow gradient descent optimizer
import tensorflow as tf
import matplotlib.pyplot as plt

input_layer = [1,2,3,4,5,6]
output_layer = [3,6,9,12,15,18]

w = tf.Variable(-1000.0)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

h = w*x

cost = tf.reduce_mean(tf.square(h-y))

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
weight_list = []

for step in range(100):
    cost_value, weight_value, _ =  sess.run([cost, w, train], feed_dict= {x:input_layer, y: output_layer})
    print("step: {0}, weight: {1}, cost: {2}".format(step, weight_value,cost_value ))
    cost_list.append(cost_value)
    weight_list.append(weight_value)

plt.plot(weight_list, cost_list)
plt.show()
