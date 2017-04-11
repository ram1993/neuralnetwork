# logistic regression using tensorflow
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

h = tf.sigmoid(tf.matmul(x,w)+b)

cost = -tf.reduce_mean(y*tf.log(h) + (1-y)*tf.log(1-h))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

Prediction = tf.cast(h>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Prediction, y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        accuracy_value, weight_value, cost_value, _ = sess.run([accuracy, w, cost, train], feed_dict={x:x_data, y:y_data})
        print("step: {0}, accuracy: {1}".format(step, accuracy_value))

    hypothesis, Prediction_value,  accuracy_value = sess.run([h,Prediction, accuracy], feed_dict={x:x_data, y:y_data})
    print(hypothesis,"\n",Prediction_value,"\n",accuracy_value)
