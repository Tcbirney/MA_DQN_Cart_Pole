import sys
sys.path.append('..')

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from nn import deep_q_nn, LossHistory
import time


# import cart_pendulum_upright
from game_envs import cart_pendulum
import numpy as np
import random
import csv
import os.path
import pygame

from copy import deepcopy


NUM_INPUT = 4 # state space dim
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.




def process_minibatch2(minibatch, target_network):
    # by Microos, improve this batch processing function 
    #   and gain 50~60x faster speed (tested on GTX 1080)
    #   significantly increase the training FPS
    
    # instead of feeding data to the model one by one, 
    #   feed the whole batch is much more efficient
    mb_len = len(minibatch)

    old_states = np.zeros(shape=(mb_len, 4))
    actions = np.zeros(shape=(mb_len,), dtype=int)
    rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, 4))


    for i, m in enumerate(minibatch):
        old_state_m, action_m, reward_m, new_state_m = m
        old_states[i, :] = old_state_m[...]
        actions[i] = action_m
        rewards[i] = reward_m
        new_states[i, :] = new_state_m[...]
        
    old_qvals = target_network.predict(old_states, batch_size=mb_len, verbose = 0)
    new_qvals = target_network.predict(new_states, batch_size=mb_len, verbose = 0)

    maxQs = np.max(new_qvals, axis=1)

    y = old_qvals
    non_term_inds = np.where(rewards != -500)[0]
    term_inds = np.where(rewards == -500)[0]

    y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (GAMMA * maxQs[non_term_inds])
    y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]

    X_train = old_states
    y_train = y
    return X_train, y_train


def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])



def log_results(filename, loss_log):
    with open('../results/single-agent/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)



def train_net(q_network, target_network, params, headless = False):

    filename = params_to_filename(params)

    observe = 1000  # Number of frames to observe before training.
    epsilon = 1
    train_frames = 10000  # Number of frames to play.
    # train_frames = 100000  # Number of frames to play.
    batchSize = params['batchSize']
    buffer = params['buffer']

    data_collect = []
    replay = []  # stores tuples of (S, A, R, S').

    loss_log = []

    # Create a new game instance.
    # game_state = cart_pendulum.GameInstance()
    game_state = cart_pendulum.GameInstance(headless)

    # Get initial state by doing nothing and getting the state.
    
    state = None
    for i in range(10):
        _, state = game_state.frame_step("None")

    t = 0
    
    while t < train_frames:
        
        
        t += 1

        if t%100 == 0:
            print(f"t: {t}")

        # Choose an action.
        if random.random() < epsilon or t < observe:
            action = np.random.randint(0, 2)  # random
        else:
            # Get Q values for each action.
            qval = q_network.predict(state, batch_size=1, verbose = 0)
            action = (np.argmax(qval))  # best


        # Take action, observe new state and get our treat.
        reward, new_state = game_state.frame_step(action)

        # Experience replay storage.
        replay.append((deepcopy(state), deepcopy(action), deepcopy(reward), deepcopy(new_state)))



    #     # If we're done observing, start training.
        if t > observe:
            
            
            if t%100 == 0:
                # predictor_model.save_weights('saved-models/temp.h5',overwrite=True)
                for target_layer, source_layer in zip(target_network.layers, q_network.layers):
                    target_layer.set_weights(source_layer.get_weights())
                print("updating target network")
            # pygame.display.quit()
            # pygame.quit()
            
            # return replay

            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)

            # Randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)

            # Get training values.
            X_train, y_train = process_minibatch2(minibatch, target_network)

            history = LossHistory()
            q_network.fit(
                X_train, y_train, batch_size=batchSize,
                epochs=1, verbose = 0, callbacks = [history])
            loss_log.append(history.losses)

        # Update the starting state with S'.
        state = new_state

        # Decrement epsilon over time.
        if epsilon > 0.1 and t > observe:
            epsilon -= (1.0/train_frames)


        # if we've crashed off of the platform then reset
        if reward == -500:
            pygame.display.quit()
            pygame.quit()
            game_state = cart_pendulum.GameInstance(headless)
            for i in range(10):
                game_state.frame_step("None")


        # Save the model every 25,000 frames.
        if t % int(train_frames/4) == 0:
            # try:
            q_network.save_weights('../models/single-agent-saved-models/' + filename + '-' +
                            str(t) + '.h5',
                            overwrite=True)
            print("Saving model %s - %d" % (filename, t))

        # except:
        # pass

    # Log results after we're done all frames.
    log_results(filename, loss_log)


nn_param = [128, 128]
params = {
    "batchSize": 64,
    "buffer": 10000,
    "nn": nn_param
}
q_network = deep_q_nn(NUM_INPUT, params['nn'])
target_network = deep_q_nn(NUM_INPUT, params['nn'])

train_net(q_network, target_network, params, False)