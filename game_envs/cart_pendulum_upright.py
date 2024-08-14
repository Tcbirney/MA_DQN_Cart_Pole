import sys, random
random.seed(1) # make the simulation the same each time, easier to debug
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np


SCREEN_HEIGHT = 1000
SCREEN_WIDTH = 1500

FLOOR_HEIGHT = 400

CAR_WIDTH = 100
CAR_HEIGHT = 30
WHEEL_RADIUS = 25

PENDULUM_LENGTH = 300
PENDULUM_WIDTH = 20
PENDULUM_RADIUS = 30


class GameInstance:
        
    def __init__(self, headless = False):
        
        self.headless = headless

        if self.headless == False:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Cart and Pendulum Sim")
        
        self.clock = pygame.time.Clock()

        

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 900.0)

        floor = self.add_floor(self.space)
        self.car, self.pendulum = self.add_car(self.space)
        # self.pendulum.body.angle = 1
        
        self.handler_car_wheel = self.space.add_collision_handler(1, 2)
        self.handler_car_pendulum = self.space.add_collision_handler(1, 3)
        self.handler_wheel_pendulum = self.space.add_collision_handler(2, 3)
        self.handler_pendulum_floor = self.space.add_collision_handler(3, 4)
        # self.handler_pend_floor = self.space.add_collision_handler(5, 4)


        self.handler_car_wheel.begin = lambda arbiter, space, data: False
        self.handler_car_pendulum.begin = lambda arbiter, space, data: False
        self.handler_wheel_pendulum.begin = lambda arbiter, space, data: False
        self.handler_pendulum_floor.begin = lambda arbiter, space, data: False
        # self.handler_pend_floor.begin = lambda arbiter, space, data: False



    def add_floor(self, space):

        floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        floor_body.position = (SCREEN_WIDTH/2, SCREEN_HEIGHT - FLOOR_HEIGHT)
        floor = pymunk.Segment(floor_body, (-SCREEN_WIDTH/2, 0), (SCREEN_WIDTH/2, 0.0), 5.0)
        floor.friction = 1
        floor.mass = 8 
        floor.collision_type = 4

        self.space.add(floor, floor_body)
        return floor


    def add_car(self, space):

        # create the car body
        car_body = pymunk.Body()
        car_body.position = (SCREEN_WIDTH/2, SCREEN_HEIGHT - FLOOR_HEIGHT - WHEEL_RADIUS - CAR_HEIGHT/2)

        car = pymunk.Poly(car_body, 
                        [(-CAR_WIDTH/2, -CAR_HEIGHT/2),
                        (-CAR_WIDTH/2, CAR_HEIGHT/2),
                        (CAR_WIDTH/2, -CAR_HEIGHT/2),
                        (CAR_WIDTH/2, CAR_HEIGHT/2)])

        car.friction = 1
        car.mass = 10
        car.color = [255, 0, 0, 255]
        car.collision_type = 1

        # create the wheel elements
        w1_body = pymunk.Body()
        w2_body = pymunk.Body()

        w1_body.position = (
            SCREEN_WIDTH/2 - CAR_WIDTH/2,
            SCREEN_HEIGHT - FLOOR_HEIGHT - WHEEL_RADIUS - CAR_HEIGHT/2)

        w2_body.position = (
            SCREEN_WIDTH/2 + CAR_WIDTH/2, 
            SCREEN_HEIGHT - FLOOR_HEIGHT - WHEEL_RADIUS - CAR_HEIGHT/2)

        w1 = pymunk.Circle(w1_body, WHEEL_RADIUS)
        w2 = pymunk.Circle(w2_body, WHEEL_RADIUS)

        w1.friction = 1
        w2.friction = 1
        w1.mass = 1 
        w2.mass = 1
        w1.collision_type = 2
        w2.collision_type = 2

        # add rotation to the wheels
        w1_rotation_joint = pymunk.PivotJoint(
            car_body, w1_body, 
            w1_body.position
        ) 

        w2_rotation_joint = pymunk.PivotJoint(
            car_body, w2_body, 
            w2_body.position
        )


        # create the pendulum
        pendulum_body = pymunk.Body()
        pendulum_body.position = (SCREEN_WIDTH/2, SCREEN_HEIGHT - FLOOR_HEIGHT - WHEEL_RADIUS - CAR_HEIGHT/2)

        pendulum_rod = pymunk.Poly(pendulum_body, 
                                [(-PENDULUM_WIDTH/2, 0),
                                    (PENDULUM_WIDTH/2, 0),
                                    (-PENDULUM_WIDTH/2, -PENDULUM_LENGTH),
                                    (PENDULUM_WIDTH/2, -PENDULUM_LENGTH)])

        pendulum_rod.friction = 1
        pendulum_rod.mass = 5

        pendulum_rotation_joint = pymunk.PivotJoint(
            car_body, pendulum_body, 
            pendulum_body.position
        ) 

        pendulum_rod.collision_type = 3

        # pendulum_end = pymunk.Circle(pendulum_body, PENDULUM_RADIUS, (0, PENDULUM_LENGTH))
        # pendulum_end.friction = 1
        # pendulum_end.mass = 5
        # pendulum_end.collision_type = 5


        self.space.add(car, car_body, 
                w1, w1_body, 
                w2, w2_body, 
                w1_rotation_joint, w2_rotation_joint,
                pendulum_rod, pendulum_body,
                pendulum_rotation_joint)

        return car, pendulum_rod


    def frame_step(self, action):

        if action == 0:
            v = self.car.body.velocity
            self.car.body.velocity = (min(0,v[0])-200, 0)
        elif action == 1:
            v = self.car.body.velocity
            self.car.body.velocity = (max(0,v[0])+200, 0)

        if self.headless == False:
            draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.screen.fill((255,255,255))

            self.space.debug_draw(draw_options)
            pygame.display.flip()
            
        self.space.step(1/50.0)
        self.clock.tick(50)


        angle = round(self.pendulum.body.angle * 180/np.pi)
        angle = angle %360
        angular_vel = round(self.pendulum.body.angular_velocity, 5)
        car_pos = round(self.car.body.position[0])
        car_vel = round(self.car.body.velocity[0])
        
        state = [angle, angular_vel, car_pos, car_vel]
        if angle >=0 and angle <= 30 or \
           angle >=330 and angle <= 360:
            reward = np.cos(angle*np.pi/180)
        else:
            reward = 0
            

        # reward = reward - angular_vel

        # print(f"State: {state}, Reward: {reward}")
        state = np.array([state])
        print(state)
        # print(reward)

        return reward, state




def main():
    pygame.init()

    game_instance = GameInstance(headless=False)

    while True:

        had_event = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit(0)
                elif event.key == pygame.K_a:
                    game_instance.frame_step(0)
                    had_event = True
                elif event.key == pygame.K_d:
                    game_instance.frame_step(1)
                    had_event = True

        if had_event == False:
            game_instance.frame_step('None')

        if game_instance.car.body.position[0] < 0 or game_instance.car.body.position[0] > SCREEN_WIDTH:
            pygame.display.quit()
            pygame.quit()
            game_instance = GameInstance(False)


if __name__ == '__main__':
    main()