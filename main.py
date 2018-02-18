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


def load_data():
    train_X = np.loadtxt('output/Othello.01e4.ggf_white_white_X.txt', delimiter=' ', dtype='float')
    train_Y = np.loadtxt('output/Othello.01e4.ggf_white_white_Y.txt', delimiter=' ', dtype='float')
    return train_X, train_Y


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y")
    return X, Y


def initialize_parameters(seed):
    tf.set_random_seed(seed)  # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [100, 64], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b1 = tf.get_variable("b1", [100, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [150, 100], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b2 = tf.get_variable("b2", [150, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [100, 150], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b3 = tf.get_variable("b3", [100, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [80, 100], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b4 = tf.get_variable("b4", [80, 1], initializer=tf.zeros_initializer())
    W5 = tf.get_variable("W5", [65, 80], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b5 = tf.get_variable("b5", [65, 1], initializer=tf.zeros_initializer())

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


def forward_propagation(X, parameters):
    # フォワードプロパゲーション
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']

    # calculate neural network
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5, A4), b5)

    return Z5


def compute_cost(Z5, Y):
    logits = tf.transpose(Z5)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost



def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=150, minibatch_size=32, print_cost=True):

    ops.reset_default_graph()
    tf.set_random_seed(1)
    costs = []

    X, Y = create_placeholders(64, 65)
    parameters = initialize_parameters(1)
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    # Use AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    print(X_train.shape)
    print(Y_train.shape)
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.
            _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train.transpose(), Y: Y_train.transpose()})

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        return parameters


(X_train, Y_train) = load_data()
parameters = model(X_train, Y_train, "", "")





