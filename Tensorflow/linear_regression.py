# linear regression using tensorflow
import tensorflow as tf
tf.set_random_seed(2)

#imput layer
x = [1,2,3,4,5,6,7,8,9]
#output layer
y = [2,4,6,8,10,12,14,16,18]

#weight vector betwwen imput and output layer
w = tf.Variable(tf.random_normal([1]), name="weight")
#bias vector for output layer
b = tf.Variable(tf.random_normal([1]), name="bias")

#hypothesis function H = X*W + B
h = x*w + b

cost = tf.reduce_mean(tf.square(h-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#start the  graph in the session and initialize all global varibale
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#training
for step in range(5000):
    sess.run(train)
    print("step : {0}, cost : {1}, weight : {2}, bias : {3}".format(step, sess.run(cost), sess.run(w), sess.run(b)))

sess.close()
