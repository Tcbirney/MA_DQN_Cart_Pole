import sys
sys.path.append('..')

from game_envs.cart_pendulum_upright import GameInstance, SCREEN_WIDTH
import numpy as np
from nn import deep_q_nn
import pygame
import datetime

NUM_SENSORS = 4

def is_fail(state):
    
    pos = state[0][2]  
    angle = state[0][0]
  
    # restart if we run off the platform
    if pos < 0 or pos > SCREEN_WIDTH:
        return 'fall'
    # restart if the pendulum falls
    elif angle > 30 and angle < 330:
        return 'drop'
    else:
        return None
        


def play(model):

    travel_distances = []
    game_state = GameInstance()

    # Do nothing to get initial.
    for i in range(10):
        _, state = game_state.frame_step(None)
    
    # Move until we fail
    while True:
        
        # check to see if we have failed, and if so return whether or not it 
        # was due to the cart running off the track or the cart dropping the ball
        fail_type = is_fail(state)
        if fail_type is not None:
                    
            # write travel logs if they cover at least the first 10 seconds of game time
            # just to make sure we have viable data present to look at
            if len(travel_distances) >= 500:
                with open('../results/single_agent_upright/travel_logs.csv', 'w') as outfile:
                    outfile.writelines(travel_distances)
            
            pygame.display.quit()
            pygame.quit()
            
            return fail_type, len(travel_distances)/50
        
        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))

        # Take action.
        _, state = game_state.frame_step(action)
        new_pos = state[0][2]
        pos_diff = str(new_pos - SCREEN_WIDTH/2) + '\n'
        travel_distances.append(pos_diff)
        
    
        

if __name__ == "__main__":
    saved_model = '../models/single-agent-upright-saved-models/128-128-64-10000-100000.h5'
    model = deep_q_nn(NUM_SENSORS, [128, 128], saved_model)
    
        
    # keep track of the times and reasons for failures
    failures = []

    
    for i in range(10):
        fail_type, duration = play(model)