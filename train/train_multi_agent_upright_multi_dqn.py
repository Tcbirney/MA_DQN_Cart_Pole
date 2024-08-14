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


time


from game_envs import multi_agent_cart_upright
# import cart_pendulum

def process_minibatch2(minibatch, target_network):
    # by Microos, improve this batch processing function 
    #   and gain 50~60x faster speed (tested on GTX 1080)
    #   significantly increase the training
    
    # instead of feeding data to the model one by one, 
    #   feed the whole batch is much more efficient
    mb_len = len(minibatch)

    old_states = np.zeros(shape=(mb_len, 8))
    actions = np.zeros(shape=(mb_len,), dtype=int)
    rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, 8))


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
    non_term_inds = np.where(rewards != 0)[0]
    term_inds = np.where(rewards == 0)[0]

    y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (GAMMA * maxQs[non_term_inds])
    y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]

    X_train = old_states
    y_train = y
    return X_train, y_train


def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])



def log_results(filename, loss_log_1, loss_log_2):

    with open('../results/multi-agent-upright-multi-dqn-cart1/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log_1:
            wr.writerow(loss_item)

    with open('../results/multi-agent-upright-multi-dqn-cart2/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log_2:
            wr.writerow(loss_item)



def train_net(cart1_q_network, cart1_target_network, cart2_q_network, cart2_target_network, params, headless = False):

    filename = params_to_filename(params)

    observe = 1000  # Number of frames to observe before training.
    epsilon = 1
    train_frames = 10000  # Number of frames to play.
    # train_frames = 100000  # Number of frames to play.
    batchSize = params['batchSize']
    buffer = params['buffer']

    data_collect = []
    cart1_replay = []  # stores tuples of (S, A, R, S').
    cart2_replay = []  # stores tuples of (S, A, R, S').

    loss_log_1 = []
    loss_log_2 = []

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
            cart1_action = np.random.randint(0, 2)  # random
            cart2_action = np.random.randint(0, 2)  # random
        else:
            # Get Q values for each action.
            cart1_qval = cart1_q_network.predict(state, batch_size=1, verbose = 0)
            cart1_action = (np.argmax(cart1_qval))  # best
            
            cart2_qval = cart2_q_network.predict(state, batch_size=1, verbose = 0)
            cart2_action = (np.argmax(cart2_qval))  # best

        # Take action, observe new state and get our treat.
        cart1_reward, cart2_reward, new_state = game_state.frame_step([cart1_action, cart2_action])

        print(f"Cart 1 Reward: {cart1_reward}\t Cart 2 Reward: {cart2_reward}")

        # Experience replay storage.
        cart1_replay.append((deepcopy(state), deepcopy(cart1_action), deepcopy(cart1_reward), deepcopy(new_state)))
        cart2_replay.append((deepcopy(state), deepcopy(cart2_action), deepcopy(cart2_reward), deepcopy(new_state)))


    #     # If we're done observing, start training.
        if t > observe:
            
            if t%10 == 0:
                # predictor_model.save_weights('saved-models/temp.h5',overwrite=True)
                for target_layer, source_layer in zip(cart1_target_network.layers, cart1_q_network.layers):
                    target_layer.set_weights(source_layer.get_weights())
                for target_layer, source_layer in zip(cart2_target_network.layers, cart2_q_network.layers):
                    target_layer.set_weights(source_layer.get_weights())
                # print("updating target network")
                

            # If we've stored enough in our buffer, pop the oldest.
            if len(cart1_replay) > buffer:
                cart1_replay.pop(0)
            if len(cart2_replay) > buffer:
                cart2_replay.pop(0)

            # Randomly sample our experience replay memory
            cart1_minibatch = random.sample(cart1_replay, batchSize)
            cart2_minibatch = random.sample(cart2_replay, batchSize)

            # Get training values.
            cart1_X_train, cart1_y_train = process_minibatch2(cart1_minibatch, cart1_target_network)
            cart2_X_train, cart2_y_train = process_minibatch2(cart2_minibatch, cart2_target_network)

            history_1 = LossHistory()
            cart1_q_network.fit(
                cart1_X_train, cart1_y_train, batch_size=batchSize,
                epochs=1, verbose = 0, callbacks = [history_1])
            loss_log_1.append(history_1.losses)
            
            history_2 = LossHistory()
            cart2_q_network.fit(
                cart2_X_train, cart2_y_train, batch_size=batchSize,
                epochs=1, verbose = 0, callbacks = [history_2])
            loss_log_2.append(history_2.losses)

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
                cart1_q_network.save_weights('../models/multi-agent-upright-multi-dqn-cart1/' + filename + '-' +
                                str(t) + '.h5',
                                overwrite=True)
                cart2_q_network.save_weights('../models/multi-agent-upright-multi-dqn-cart2/' + filename + '-' +
                                str(t) + '.h5',
                                overwrite=True)
            
                print("Saving model %s - %d" % (filename, t))
                
            except:
                pass


    # Log results after we're done all frames.
    log_results(filename, loss_log_1, loss_log_2)




# def launch_learn(params):
#     filename = params_to_filename(params)
#     print("Trying %s" % filename)
#     # Make sure we haven't run this one.
#     if not os.path.isfile('results/cart_pendulum/loss_data-' + filename + '.csv'):
#         # Create file so we don't double test when we run multiple
#         # instances of the script at the same time.
#         open('results/cart_pendulum/loss_data-' + filename + '.csv', 'a').close()
#         print("Starting test.")
#         # Train.
#         predictor_model = deep_q_nn(NUM_INPUT, params['nn'])
#         train_net(predictor_model, params)
#     else:
#         print("Already tested.")



nn_param = [128, 128]
params = {
    'train_size': 100000,
    "batchSize": 64,
    "buffer": 10000,
    "nn": nn_param
}
cart1_q_network = deep_q_nn(NUM_INPUT, params['nn'])
cart1_target_network = deep_q_nn(NUM_INPUT, params['nn'])

cart2_q_network = deep_q_nn(NUM_INPUT, params['nn'])
cart2_target_network = deep_q_nn(NUM_INPUT, params['nn'])

train_net(cart1_q_network, cart1_target_network, cart2_q_network, cart2_target_network, params, False)