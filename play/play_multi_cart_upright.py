import sys
sys.path.append('..')

from game_envs.multi_agent_cart_upright import GameInstance, SCREEN_WIDTH, CAR_OFFSET
import numpy as np
from nn import deep_q_nn
import pygame
import datetime


NUM_SENSORS = 8

cart1_travel_distances = []
cart2_travel_distances = []


def test_reset(state):
    
    reset = False
    
    cart1_pos = state[0][2]
    cart2_pos = state[0][6]    
    
    # restart if we run off the platform
    if (cart1_pos < 0 or cart1_pos > SCREEN_WIDTH) or (cart2_pos < 0 or cart2_pos > SCREEN_WIDTH):
        reset = True
        
    # restart if the pendulum falls
    cart1_angle = state[0][0]
    cart2_angle = state[0][4]
    if (cart1_angle > 30 and cart1_angle < 330) or (cart2_angle > 30 and cart2_angle < 330):
        reset = True
        
    if reset == True:
        pygame.display.quit()
        pygame.quit()
        game_state = GameInstance()
        for i in range(10):
            _, state = game_state.frame_step(None)

    return state



def play(cart1_model, cart2_model):

    game_state = GameInstance()

    # Do nothing to get initial.
    state = None
    for i in range(10):
        cart1_reward, cart2_reward, state = game_state.frame_step([2, 2])

    start_time = datetime.datetime.now()
    
    # Move for a minute
    while True and len(cart1_travel_distances) < 500:
        # Choose action.
        cart1_action = (np.argmax(cart1_model.predict(state, batch_size=1)))
        cart2_action = (np.argmax(cart2_model.predict(state, batch_size=1)))

        # Take action.
        cart1_reward, cart2_reward, state = game_state.frame_step([cart1_action, cart2_action])

        cart1_pos = state[0][2]
        cart1_pos_diff = str(cart1_pos - (SCREEN_WIDTH/2-CAR_OFFSET)) + '\n'
        cart1_travel_distances.append(cart1_pos_diff)
        
        cart2_pos = state[0][6]
        cart2_pos_diff = str(cart2_pos - (SCREEN_WIDTH/2+CAR_OFFSET)) + '\n'
        cart2_travel_distances.append(cart2_pos_diff)
    
        
    # cleanup game state
    pygame.display.quit()
    pygame.quit()
    
    # write travel logs
    with open('../results/multi-agent-upright-multi-dqn-cart1/travel_logs.csv', 'w') as outfile:
        outfile.writelines(cart1_travel_distances)

    with open('../results/multi-agent-upright-multi-dqn-cart2/travel_logs.csv', 'w') as outfile:
        outfile.writelines(cart2_travel_distances)


if __name__ == "__main__":
    cart1_saved_model = '../models/multi-agent-upright-multi-dqn-cart1/128-128-64-10000-100000.h5'
    cart2_saved_model = '../models/multi-agent-upright-multi-dqn-cart2/128-128-64-10000-100000.h5'
    cart1_model = deep_q_nn(NUM_SENSORS, [128, 128], cart1_saved_model)
    cart2_model = deep_q_nn(NUM_SENSORS, [128, 128], cart2_saved_model)
    play(cart1_model, cart2_model)