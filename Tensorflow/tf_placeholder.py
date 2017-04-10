#if we want to use tf.Placeholder insteand of tf.Variable then value of these varibale are always apaased by feed dictionary
import tensorflow as tf
tf.set_random_seed(55)

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

h = x*w + b

cost = tf.reduce_mean(tf.square(h - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(5000):
    # note that name of cost, weight, bias should be different than that of defined earlier
    cost_value, weight, bias, _ = sess.run([cost,w,b,train],feed_dict={x: [1, 2, 3], y: [1, 2, 3]} )
    print(step, cost_value, weight, bias)
