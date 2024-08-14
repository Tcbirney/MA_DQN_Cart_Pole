import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.optimizers import RMSprop

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def deep_q_nn(input_dim, params, load = ''):
    model = tf.keras.models.Sequential([
        # intput layer, 4 elements to the state space
        tf.keras.layers.Dense(units = params[0], activation='relu', kernel_initializer = "lecun_uniform"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units = params[1], activation = 'relu', kernel_initializer = "lecun_uniform"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, kernel_initializer = "lecun_uniform") # outputs the q values for each of the possible states
        # left, right, stand still
    ])
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)


    if load:
        model.build(input_shape = (1, input_dim))
        model.load_weights(load)

    return model


def deep_q_nn_multi_agent(input_dim, params, load = ''):
    model = tf.keras.models.Sequential([
        # intput layer, 4 elements to the state space
        tf.keras.layers.Dense(units = params[0], activation='relu', kernel_initializer = "lecun_uniform"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units = params[1], activation = 'relu', kernel_initializer = "lecun_uniform"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, kernel_initializer = "lecun_uniform") # outputs the q values for each of the possible states
        # left, right
    ])

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)


    if load:
        model.build(input_shape = (1, input_dim))
        model.load_weights(load)

    return model