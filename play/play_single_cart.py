import sys
sys.path.append('..')

from game_envs.cart_pendulum import GameInstance
import numpy as np
from nn import deep_q_nn
import pygame

NUM_SENSORS = 4


def play(model):

    game_state = GameInstance()

    # Do nothing to get initial.
    for i in range(10):
        _, state = game_state.frame_step(None)

    # Move.
    while True:
        # if game_state.car.body.position[0] < 0 or\
        #     game_state.car.body.position[0] > cart_pendulum_upright.SCREEN_WIDTH:
        #     pygame.display.quit()
        #     pygame.quit()
        #     game_state = cart_pendulum_upright.GameInstance()
        #     _, state = game_state.frame_step(None)
        #     continue
        
        # angle = state[0][0]
        # if (angle >=0 and angle <= 30) or \
        #    (angle >=330 and angle <= 360):
        #     pygame.display.quit()
        #     pygame.quit()
        #     game_state = cart_pendulum_upright.GameInstance()
        #     for i in range(10):
        #         _, state = game_state.frame_step(None)
        #     continue
        


        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))

        # Take action.
        _, state = game_state.frame_step(action)


if __name__ == "__main__":
    saved_model = '../models/single-agent-saved-models/128-128-64-10000-100000.h5'
    model = deep_q_nn(NUM_SENSORS, [128, 128], saved_model)
    play(model)