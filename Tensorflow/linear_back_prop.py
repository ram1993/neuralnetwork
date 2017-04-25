import tensorflow as tf
tf.set_random_seed(45)

# x_data = [[1], [2], [3], [4]]
# y_data = [[1], [2], [3], [4]]


x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]


x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(0.5)

h = tf.matmul(x, w) + b

assert h.shape.as_list() == y.shape.as_list()
diff = (h - y)


# backpropogation chain rule
d_b = diff
d_w = tf.matmul(tf.transpose(x), diff)

learning_rate = 1e-6

step = [ tf.assign(w,w- learning_rate*d_w), tf.assign(b, b- learning_rate*tf.reduce_mean(d_b))]

RMSE = tf.reduce_mean(tf.square(h -y))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    #sess.run(init)

    for i in range(50000):

        print(i, sess.run([step, RMSE], feed_dict={x:x_data, y: y_data}))

    print(sess.run(h, feed_dict={x:x_data}))
