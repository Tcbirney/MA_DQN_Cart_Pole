import sys
sys.path.append('..')

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from nn import deep_q_nn_multi_agent, LossHistory
import time


from game_envs import multi_agent_cart_upright
# import cart_pendulum
import numpy as np
import random
import csv
import os.path
import pygame

from copy import deepcopy


NUM_INPUT = 8 # state space dim
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.


def process_minibatch2(minibatch, target_network):
    # by Microos, improve this batch processing function 
    #   and gain 50~60x faster speed (tested on GTX 1080)
    #   significantly increase the training
    
    # instead of feeding data to the model one by one, 
    #   feed the whole batch is much more efficient
    mb_len = len(minibatch)

    old_states = np.zeros(shape=(mb_len, 8))
    cart1_actions = np.zeros(shape=(mb_len,), dtype=int)
    cart2_actions = np.zeros(shape=(mb_len,), dtype=int)
    cart1_rewards = np.zeros(shape=(mb_len,))
    cart2_rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, 8))


    for i, m in enumerate(minibatch):
        old_state_m, cart1_action_m, cart2_action_m, cart1_reward_m, cart2_reward_m, new_state_m = m
        old_states[i, :] = old_state_m[...]
        cart1_actions[i] = cart1_action_m
        cart2_actions[i] = cart2_action_m
        cart1_rewards[i] = cart1_reward_m
        cart2_rewards[i] = cart2_reward_m
        new_states[i, :] = new_state_m[...]
        
    old_qvals = target_network.predict(old_states, batch_size=mb_len, verbose = 0)
    new_qvals = target_network.predict(new_states, batch_size=mb_len, verbose = 0)
    
    # print(new_qvals)
    
    
    # qval arrays are mb len x 4, left half is cart 1, right half is cart 2
    cart1_maxQs = np.max(new_qvals[:, :2], axis=1)
    cart2_maxQs = np.max(new_qvals[:, 2:], axis=1)
    
    
    y = old_qvals
    
    # get the non term and term indeces for each cart given their independent rewards
    cart1_non_term_inds = np.where(cart1_rewards != 0)[0]
    cart2_non_term_inds = np.where(cart2_rewards != 0)[0]
    
    cart1_term_inds = np.where(cart1_rewards == 0)[0]
    cart2_term_inds = np.where(cart2_rewards == 0)[0]



    # print(f"old_states: {old_states.shape}")
    # print(f"new_states: {new_states.shape}")
    # print(f"cart1_action_m: {cart1_actions.shape}")
    # print(f"cart2_action_m: {cart2_actions.shape}")
    # print(f"cart1_reward_m: {cart1_rewards.shape}")
    # print(f"cart2_reward_m: {cart2_rewards.shape}")
    # print(f"old_qvals: {old_qvals.shape}")
    # print(f"new_qvals: {new_qvals.shape}")
    # print(f"cart1_maxQs: {cart1_maxQs.shape}")
    # print(f"cart2_maxQs: {cart2_maxQs.shape}")
    # print(f"y: {y.shape}")
    

    # adjust the cart 1 q values for non terminal and terminal reward indeces
    y[cart1_non_term_inds, cart1_actions[cart1_non_term_inds].astype(int)] = cart1_rewards[cart1_non_term_inds] + (GAMMA * cart1_maxQs[cart1_non_term_inds])
    y[cart1_term_inds, cart1_actions[cart1_term_inds].astype(int)] = cart1_rewards[cart1_term_inds]
    
    # because the cart 2 actions are within the range of [0,1] then we need to offset the y index by 2
    y[cart2_non_term_inds, cart2_actions[cart2_non_term_inds].astype(int) + 2] = cart2_rewards[cart2_non_term_inds] + (GAMMA * cart2_maxQs[cart2_non_term_inds])
    y[cart2_term_inds, cart2_actions[cart2_term_inds].astype(int) + 2] = cart2_rewards[cart2_term_inds]
    
    # so X_train here should be mb_len x 8
    # and y_train here should be mb_len x 4
    X_train = old_states
    y_train = y
    
    
    
    
    return X_train, y_train


def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])


def log_results(filename, loss_log):

    with open('../results/multi-agent-upright-single-dqn/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


def train_net(q_network, target_network, params, headless = False):

    filename = params_to_filename(params)

    observe = 1000  # Number of frames to observe before training.
    epsilon = 1
    train_frames = 10000  # Number of frames to play.

    batchSize = params['batchSize']
    buffer = params['buffer']

    data_collect = []
    replay = []  # stores tuples of (S, A, R, S').

    loss_log = []

    # Create a new game instance.
    # game_state = cart_pendulum.GameInstance()
    game_state = multi_agent_cart_upright.GameInstance(headless)

    # Get initial state by doing nothing and getting the state.
    
    cart1_reward, cart2_reward, state = None, None, None
    for i in range(10):
        cart1_reward, cart2_reward, state = game_state.frame_step([2, 2])

    t = 0
    
    while t < train_frames:
        
        if t%1000 == 0:
            print(f"t: {t}")

        t += 1

        # Choose an action.
        if random.random() < epsilon or t < observe:
            cart1_action = np.random.randint(0, 2)
            cart2_action = np.random.randint(0, 2)
        else:
            # Get Q values for each action.
            qval = q_network.predict(state, batch_size=1, verbose = 0)
            
            # when the model, q_val will be 1 x 4
            # split into two and get the argmax of each best action
            # each action is in range [0,1] because they are the indeces of
            # the best q val in each of the 1x2 vectors
            cart1_action = np.argmax(qval[0, :2])
            cart2_action = np.argmax(qval[0, 2:])
            

        # Take action, observe new state and get our treat.
        cart1_reward, cart2_reward, new_state = game_state.frame_step([cart1_action, cart2_action])


        # Experience replay storage.
        replay.append((deepcopy(state), deepcopy(cart1_action), deepcopy(cart2_action), deepcopy(cart1_reward),  deepcopy(cart2_reward), deepcopy(new_state)))


    #     # If we're done observing, start training.
        if t > observe:
            
            if t%10 == 0:
                # predictor_model.save_weights('saved-models/temp.h5',overwrite=True)
                for target_layer, source_layer in zip(target_network.layers, q_network.layers):
                    target_layer.set_weights(source_layer.get_weights())
                

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
        if cart1_reward == 0 or cart2_reward == 0:
            pygame.display.quit()
            pygame.quit()
            game_state = multi_agent_cart_upright.GameInstance(headless)
            for i in range(10):
                game_state.frame_step([2, 2])
                

        # Save the model every 25,000 frames.
        if t % int(train_frames/4) == 0:
            
            try:
                q_network.save_weights('../models/multi-agent-upright-single-dqn/' + filename + '-' +
                                str(t) + '.h5',
                                overwrite=True)
            
                print("Saving model %s - %d" % (filename, t))
            except:
                pass
            
    log_results(filename, loss_log)


nn_param = [128, 128]
params = {
    "batchSize": 64,
    "buffer": 10000,
    "nn": nn_param
}

q_network = deep_q_nn_multi_agent(NUM_INPUT, params['nn'])
target_network = deep_q_nn_multi_agent(NUM_INPUT, params['nn'])

train_net(q_network, target_network, params, False)