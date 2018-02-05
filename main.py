import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
# from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')
loss = tf.Variable((y - y_hat)**2, name='loss')
init = tf.global_variables_initializer()

with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y")
    return X, Y


def initialize_parameters(seed):
    tf.set_random_seed(seed)  # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [100, 64], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b1 = tf.get_variable("b1", [100, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [150, 100], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b2 = tf.get_variable("b2", [150, 1], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    W3 = tf.get_variable("W3", [100, 150], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b3 = tf.get_variable("b3", [100, 1], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    W4 = tf.get_variable("W4", [80, 100], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b4 = tf.get_variable("b4", [80, 1], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    W5 = tf.get_variable("W5", [65, 80], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b5 = tf.get_variable("b5", [65, 1], initializer=tf.contrib.layers.xavier_initializer(seed=seed))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5}

    return parameters


X, Y = create_placeholders(64*64, 65)
parameters = initialize_parameters(1)

