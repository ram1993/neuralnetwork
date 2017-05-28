# import tensorflow as tf
# import numpy as np
# from tensorflow.contrib import rnn
#
#
# idx2char = ['h', 'e','l','o']
#
# x_data = [0,1,2,2,3] # hello
# y_data = [0,3,2,3,1] # holoe
#
#
# x_one_hot = [[[1,0,0,0],
#              [0,1,0,0],
#              [0,0,1,0],
#              [0,0,1,0],
#              [0,0,0,1]]]
#
#
# batch_size= 1
# num_class = 4
# sequence_length = 5
# num_hidden = 4
# input_dim = 4
#
# X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
# Y = tf.placeholder(tf.float32, [None, sequence_length])
#
# cell = tf.contrib.rnn.BasicLSTMCell(num_units = num_hidden, state_is_tuple = True)
# initial_state = cell.zero_state(batch_size, tf.float32)
#
# output, _state = tf.nn.dynamic_rnn(cell,X, initial_state=initial_state, dtype = tf.float32)
#
# x_for_fc = tf.reshape(output, [-1, num_hidden])
#
# output = tf.contrib.layers.fully_connected(inputs = x_for_fc, num_outputs = num_class, activation_fn=None)
#
# output = tf.reshape(output, [batch_size, sequence_length, num_class])
#
# weights = tf.ones([batch_size, sequence_length])
#
# sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=output, targets=Y, weights=weights)
#
# loss = tf.reduce_mean(sequence_loss)
# train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#
# prediction = tf.argmax(outputs, axis=2)
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(50):
#         l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
#         result = sess.run(prediction, feed_dict={X: x_one_hot})
#         print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)
#
#         # print char using dic
#         result_str = [idx2char[c] for c in np.squeeze(result)]
#         print("\tPrediction str: ", ''.join(result_str))



# Lab 12 RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

idx2char = ['h', 'i', 'e', 'l', 'o']
# Teach hello: hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [[1, 0, 2, 3, 3, 4]]    # ihello

num_classes = 5
input_dim = 5  # one-hot size
hidden_size = 3  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
learning_rate = 0.1

X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label


with tf.name_scope("LSTM") as scope:
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

    #cell_hist = tf.summary.histogram("cell", cell)
    outputs_hist = tf.summary.histogram("cell_output", outputs)
    _states_hist = tf.summary.histogram("cell_states", _states)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
cost_summ = tf.summary.scalar("cost", loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/rnn_logs_r0_01")
    writer.add_graph(sess.graph) # Show the graph

    sess.run(tf.global_variables_initializer())
    for i in range(50):
        summary, l, _ = sess.run([merged_summary, loss, train], feed_dict={X: x_one_hot, Y: y_data})
        writer.add_summary(summary, global_step=i)
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
print("\tPrediction str: ", ''.join(result_str))
