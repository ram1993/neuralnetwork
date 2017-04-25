#tensorboard --logdir=./logs  command to run on terminal 127.0.0.1:6006 to view in browser
import tensorflow as tf
import numpy as np
tf.set_random_seed(234)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

x = tf.placeholder(tf.float32, shape = [None,2], name="x-input")
y = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")


with tf.name_scope("Layer1") as scope:
    w1 = tf.Variable(tf.random_normal([2,2]), name="weight1")
    b1 = tf.Variable(tf.random_normal([2]), name="bias1")
    h1 = tf.sigmoid(tf.matmul(x,w1) + b1)

    w1_hist = tf.summary.histogram("weight1", w1)
    b1_hist = tf.summary.histogram("bias1", b1)
    layer1_hist = tf.summary.histogram("layer1",h1)


with tf.name_scope("Layer2") as scope:
    w2 = tf.Variable(tf.random_normal([2,1]), name="weight2")
    b2 = tf.Variable(tf.random_normal([1]), name="bias2")
    h2 = tf.sigmoid(tf.matmul(h1,w2)+b2)

    w2_hist = tf.summary.histogram("weight2", w2)
    b2_hist = tf.summary.histogram("bias2", b2)
    layer2_hist = tf.summary.histogram("layer2",h2)

with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(y*tf.log(h2) + (1-y)*tf.log(1-h2))
    cost_hist = tf.summary.histogram("cost", cost)

with tf.name_scope("train") as scope:
    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)



prediction = tf.cast(h2>0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

accuracy_summary = tf.summary.scalar("Accuracy", accuracy)

with tf.Session() as sess:

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
    writer.add_graph(sess.graph) # Show the graph

    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        _, summary, cost_value = sess.run([train, merged_summary, cost], feed_dict={x:x_data, y:y_data})
        writer.add_summary(summary, global_step=step)
        print(step, cost_value)
        #print(step, sess.run(cost, feed_dict={x:x_data, y:y_data}))
        #sess.run([w1,w2])

    hypothesis , p, a = sess.run([h2, prediction, accuracy], feed_dict={x:x_data, y:y_data})
    print(hypothesis)
    print(p)
    print(a)
