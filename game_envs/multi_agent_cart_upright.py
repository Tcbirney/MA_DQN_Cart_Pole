import sys, random
random.seed(1) # make the simulation the same each time, easier to debug
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import numpy as np


SCREEN_HEIGHT = 800
SCREEN_WIDTH = 1500

FLOOR_HEIGHT = 250

CAR_WIDTH = 100
CAR_HEIGHT = 30
WHEEL_RADIUS = 25

PENDULUM_LENGTH = 200
PENDULUM_WIDTH = 20
PENDULUM_RADIUS = 30

FLOOR_COLLISION_TYPE = 0
CART_COLLISION_TYPE = 1
WHEEL_COLLISION_TYPE = 2
PENDULUM_COLLISION_TYPE = 3

CAR_OFFSET = 400


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

        self.car1, self.pendulum1, self.car1_w1, self.car2_w2 = self.add_car(self.space, -CAR_OFFSET, 0)
        self.car2, self.pendulum2, self.car1_w1, self.car2_w2 = self.add_car(self.space, CAR_OFFSET, 10)

        spring_constraint = pymunk.DampedSpring(
            self.car1.body, self.car2.body, 
            (0,0), (0,0),
            CAR_OFFSET*2, 5, 0.5
        )
        self.space.add(spring_constraint)


    def add_floor(self, space):

        floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        floor_body.position = (SCREEN_WIDTH/2, SCREEN_HEIGHT - FLOOR_HEIGHT)
        floor = pymunk.Segment(floor_body, (-SCREEN_WIDTH/2, 0), (SCREEN_WIDTH/2, 0.0), 5.0)
        floor.friction = 1
        floor.mass = 8 
        floor.collision_type = FLOOR_COLLISION_TYPE

        self.space.add(floor, floor_body)
        return floor


    def add_car(self, space, x_pos, robot_id):

        # create the car body
        car_body = pymunk.Body()
        car_body.position = (SCREEN_WIDTH/2 + x_pos, SCREEN_HEIGHT - FLOOR_HEIGHT - WHEEL_RADIUS - CAR_HEIGHT/2)

        car = pymunk.Poly(car_body, 
                        [(-CAR_WIDTH/2, -CAR_HEIGHT/2),
                        (-CAR_WIDTH/2, CAR_HEIGHT/2),
                        (CAR_WIDTH/2, -CAR_HEIGHT/2),
                        (CAR_WIDTH/2, CAR_HEIGHT/2)])

        car.friction = 1
        car.mass = 10
        car.color = [255, 0, 0, 255]
        car.collision_type = CART_COLLISION_TYPE + robot_id

        # create the wheel elements
        w1_body = pymunk.Body()
        w2_body = pymunk.Body()

        w1_body.position = (
            SCREEN_WIDTH/2 + x_pos - CAR_WIDTH/2,
            SCREEN_HEIGHT - FLOOR_HEIGHT - WHEEL_RADIUS - CAR_HEIGHT/2)

        w2_body.position = (
            SCREEN_WIDTH/2 + x_pos + CAR_WIDTH/2, 
            SCREEN_HEIGHT - FLOOR_HEIGHT - WHEEL_RADIUS - CAR_HEIGHT/2)

        w1 = pymunk.Circle(w1_body, WHEEL_RADIUS)
        w2 = pymunk.Circle(w2_body, WHEEL_RADIUS)

        w1.friction = 1
        w2.friction = 1
        w1.mass = 1 
        w2.mass = 1
        w1.collision_type = WHEEL_COLLISION_TYPE + robot_id
        w2.collision_type = WHEEL_COLLISION_TYPE + robot_id

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
        pendulum_body.position = (SCREEN_WIDTH/2 + x_pos, SCREEN_HEIGHT - FLOOR_HEIGHT - WHEEL_RADIUS - CAR_HEIGHT/2)

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

        pendulum_rod.collision_type = PENDULUM_COLLISION_TYPE + robot_id

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


        
        handler_car_wheel = self.space.add_collision_handler(
            CART_COLLISION_TYPE + robot_id, WHEEL_COLLISION_TYPE + robot_id)
        handler_car_pendulum = self.space.add_collision_handler(
            CART_COLLISION_TYPE + robot_id, PENDULUM_COLLISION_TYPE + robot_id)
        handler_wheel_pendulum = self.space.add_collision_handler(
            WHEEL_COLLISION_TYPE + robot_id, PENDULUM_COLLISION_TYPE + robot_id)
        handler_pendulum_floor = self.space.add_collision_handler(
            PENDULUM_COLLISION_TYPE + robot_id, FLOOR_COLLISION_TYPE)


        handler_car_wheel.begin = lambda arbiter, space, data: False
        handler_car_pendulum.begin = lambda arbiter, space, data: False
        handler_wheel_pendulum.begin = lambda arbiter, space, data: False
        handler_pendulum_floor.begin = lambda arbiter, space, data: False


        return car, pendulum_rod, w1, w1



    def get_reward_state(self, car, pendulum):

        angle = round(pendulum.body.angle * 180/np.pi)
        angle = angle %360
        angular_vel = round(pendulum.body.angular_velocity, 5)
        car_pos = round(car.body.position[0])
        car_vel = round(car.body.velocity[0])
        
        state = [angle, angular_vel, car_pos, car_vel]
        if angle >=0 and angle <= 30 or \
           angle >=330 and angle <= 360:
            reward = np.cos(angle*np.pi/180)
           
        else:
            reward = 0


        return reward, state


    def frame_step(self, actions):

        if actions[0]== 0: # car 1 left
            v = self.car1.body.velocity
            self.car1.body.velocity = (min(0,v[0])-200, 0)
        elif actions[0] == 1: # car 1 right
            v = self.car1.body.velocity
            self.car1.body.velocity = (max(0,v[0])+200, 0)
            
        if actions[1] == 0: # car 2 left
            v = self.car2.body.velocity
            self.car2.body.velocity = (min(0,v[0])-200, 0)
        elif actions[1] == 1: # car 2 right
            v = self.car2.body.velocity
            self.car2.body.velocity = (max(0,v[0])+200, 0)



        if self.headless == False:
            draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.screen.fill((255,255,255))

            self.space.debug_draw(draw_options)
            pygame.display.flip()
            
        self.space.step(1/50.0)
        self.clock.tick(50)
        
        

        reward1, state1 = self.get_reward_state(self.car1, self.pendulum1)
        reward2, state2 = self.get_reward_state(self.car2, self.pendulum2)
        
        state = state1 + state2
        state = np.array([state])
        # print(state)

        return reward1, reward2, state




def main():
    pygame.init()

    game_instance = GameInstance(False)

    while True:

        had_event = False

        actions = [2,2]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit(0)
                if event.key == pygame.K_a:
                    had_event = True
                    actions[0] = 0
                if event.key == pygame.K_d:
                    had_event = True
                    actions[0] = 1
                if event.key == pygame.K_j:
                    had_event = True
                    actions[1] = 0
                if event.key == pygame.K_l:
                    had_event = True
                    actions[1] = 1

                game_instance.frame_step(actions)


        if had_event == False:
            game_instance.frame_step('None')


        fail = False

        if game_instance.car1.body.position[0] < 0 + CAR_WIDTH/2\
            or game_instance.car1.body.position[0] > SCREEN_WIDTH - CAR_WIDTH/2:
            fail = True
        if game_instance.car2.body.position[0] < 0 + CAR_WIDTH/2\
            or game_instance.car2.body.position[0] > SCREEN_WIDTH - CAR_WIDTH/2:
            fail = True

        if fail == True:
            pygame.display.quit()
            pygame.quit()
            game_instance = GameInstance(False)


if __name__ == '__main__':
    main()