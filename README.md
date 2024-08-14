# ENPM 690: Cart Pole Project

### Game Envs

I have built multiple versions of the cart pole game environment
- a single agent cart pole where the pole is in a downward position
- a single agent cart pole where the pole is in an upright position
- a multi agent cart pole where the poles are upright and the carts are connected with a spring

In the environments where the poles are starting in the upright position, the rewards are equal to the cosine of the pole angle, unless they are 30 degrees away from the upright angle, in which case the reward is zero.

In the environment where the pole is starting in a downward position, the reward is also the cosine of the pole angle, unless the angle is within 20 degrees of the upright position, in which case it is subtracted by 0.0001*angular_velocity^2. this is done to encourage slow angular velocities once the pole is close to being upright. If the cart moves off the track, then the reward is -500 regardless of any other state condition.

Each environment reports its state and reward. In the single agent environments the state is 
```[angle, angular velocity, cart position, cart velocity]```

But in the multi agent environment, the state represents the state of each cart.
```[angle 1, angular velocity 1, cart 1 position cart 1 velocity, angle 2, angular velocity 2, cart 2 position cart 2 velocity]```


### Trainers

I have also built trainers for each of these environments. The single agent environments both use one policy network and one target network. After a certain value c, the policy network weights are copied over to the target network. These trainers run through 100,000 iterations and take about three hours to complete training. Model weights are saved to a file every 25,000 iterations.

I have created two trainers for the multi agent environments. In each model the state in the action replay consists of the state of each cart. 

The first trainer uses an architecture where each cart has their own policy and target network. Each of these netwrks ingests the state of both carts, but also only ingests the actions and rewards of the cart it belongs to. 

In the second trainer, I have attempted to implement an architecture where a single set of policy and target networks can control both carts at the same time. Here the action replay consists of the states, actions, and rewards of both robots. The neural network also has an output layer twice the size so they can predict the Q-Values for both carts. 

### Performance

The environment where the pole must swing up does not perform as expected at all. I think this may have something to do with the reward I set, the batch size, or the minibatch size. However, the cart pole system where the only goal is to maintain the pole position is working well and, as far as I have observed, can support the pole indefinitely.

The multi agent environment with two sets of policy and target networks is working somewhat well. It is able to hold both of the poles up but may occasionally fall.

The multi agent environment which only uses one policy network and one target network does not perform well at all. I think this could be due to a number of factors such as

- batch size
- learning rate
- network architecture complexity
- reward function

## Next Steps

I would like to experiment with batch sizes and learning rates to see if I can converge to a solution faster. I think 100,000 turns is too much and is slowing down development. I would also like to implement a system for loss tracking as I do not have that yet.

I would also like to experiment more with the reward function, especially in the multi agent environments because it could be possible to communicate information between the two carts using their rewards.

## Instructions

To run any of the simulations, cd into the play directory and run any of the python files.

To change what saved model you want to use, change the file directory in the play script to the model you want to use. All saved models use the same hyperparameters, and only differ in the number of timesteps each were trained on.

## Dependencies

- pymunk
- pygame
- numpy
- tensorflow
- keras