import random
import math
import pygame
from pygame.color import THECOLORS

import pymunk

from flatland.entities.agent import Agent
from flatland.entities.edible import Edible
from flatland.entities.obstacle import Obstacle
from PIL import Image  # Pillow package

import numpy as np


class Env(object):

    def __init__(self, **kwargs):
        """
        Instantiate a game with the given parameters
        :param horizon: int, time horizon of an episode
        :param done: bool, True if the episode is terminated
        :param mode: 'goal' or 'health':
            - if 'goal' : we use the field goal to create a goal and the simulation ends
                when the goal is reached or when we reach the horizon
            - if 'survival', the health measurements in initialized to 100 and the simulation
                ends when the health reaches 0 or when we reach the horizon
        :param shape: size 2 tuple with height and width of the environment
        :param goal: dict with the following fields, only useful if mode is 'goal'
            - size: float, size of the goal
            - position: size 2 tuple giving the position or 'random'
        :param walls: dict with the following fields:
            - number: int, number of walls in the environment
            - size: float, size of the walls
            - position: array of coordinates or 'random'
        :param poisons: dict with the following fields
            - number: int, number of poisons in the environment
            - size: float, size of the poisons
            - reap: bool, whether another poison object reappears when one is consumed
        :param fruits: dict with the following fields
             - number: int, number of fruits in the environment
             - size: float, size of the fruits
             - reap: bool, whether another fruit object reappears when one is consumed
        :param agent: the agent evolving in the environment
        :param display: bool, whether to display the task or not
        """

        # Save the arguments for reset
        self.parameters = kwargs

        self.done = False
        self.t = 0
        self.horizon = kwargs['horizon']
        self.width, self.height = kwargs['shape']
        self.display = kwargs['display']
        if self.display:
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.screen.set_alpha(None)
        else:
            self.screen = pygame.Surface((self.width, self.height))
            self.screen.set_alpha(None)
        self.clock = pygame.time.Clock()
        
        self.npimage = np.zeros((self.width, self.height, 3))

        # Set a surface to compute Sensors

        # Initialize pymunk space
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        self.space.collision_slop = 0
        self.space.collision_persistence = 1
        self.space.collision_bias = 0
        self.handle_collisions()

        # Define the external walls
        texture_params = kwargs['walls_texture']
        self.obstacles = [
            Obstacle(
                shape='rectangle',
                position=(self.width/2, 5),
                angle=0,
                texture=texture_params,
                environment=self,
                length=self.width,
                width=10,
            ),
            Obstacle(
                shape='rectangle',
                position=(self.width / 2, self.height-5),
                angle=0,
                texture=texture_params,
                environment=self,
                length=self.width,
                width=10,
            ),
            Obstacle(
                shape='rectangle',
                position=(5, self.height / 2),
                angle=math.pi/2,
                texture=texture_params,
                environment=self,
                length=self.height,
                width=10,
            ),
            Obstacle(
                shape='rectangle',
                position=(self.width - 5, self.height / 2),
                angle=math.pi/2,
                texture=texture_params,
                environment=self,
                length=self.height,
                width=10,
            )
        ]

        # Add obstacles
        for obstacle_params in kwargs['obstacles']:
            obstacle_params['environment'] = self
            obstacle = Obstacle(**obstacle_params)
            self.obstacles.append(obstacle)

        # Define the episode mode
        self.mode = kwargs['mode']

        # Add the agent
        self.agent = Agent(environment=self, **kwargs['agent'])
        
        self.agent.update_state()

    def reload_screen(self, display):
        # Fill the screen
        self.screen.fill(THECOLORS["black"])
        
        # Do 10 mini-timesteps in pymunk for 1 timestep in our environment
        for _ in range(10):
            self.space.step(1. / 10)
        
        # Draw the entities
        self.drawEnvironment()
        
        # Get top view image of environment
        # remark: very slow
        data = pygame.image.tostring(self.screen, 'RGB')
        pil_image = Image.frombytes('RGB', (self.width, self.height), data)
        image = np.asarray(pil_image.convert('RGB'))
        self.npimage = image
                        
        # Draw the agent
        self.agent.draw()
        
        # Update the display
        if display:
            pygame.display.flip()
        self.clock.tick()

    def handle_collisions(self):

        def begin_fruit_collision(arbiter, space, *args, **kwargs):

            # Remove the previous shape
            shapes = arbiter.shapes
            for shape in shapes:
                if shape.collision_type == 2:
                    self.fruits.remove(shape.body.entity)
                    space.remove((shape, shape.body))

                    # Update the measurements
                    self.agent.update_meas('items', 1)
                    self.agent.update_health(shape.body.entity.reward, self.mode)
                    self.agent.update_meas('fruits', 1)
                    self.agent.reward += shape.body.entity.reward

            if self.fruit_params['respawn']:
                self.fruits.append(Edible(**self.fruit_params))

            return False

        def begin_poison_collision(arbiter, space, *args, **kwargs):

            # Remove the previous shape
            shapes = arbiter.shapes
            for shape in shapes:
                if shape.collision_type == 3:
                    self.poisons.remove(shape.body.entity)
                    space.remove((shape, shape.body))

                    # Update the measurements
                    self.agent.update_meas('items', 1)
                    self.agent.update_health(shape.body.entity.reward, self.mode)
                    self.agent.update_meas('poisons', 1)
                    self.agent.reward += shape.body.entity.reward

            if self.poison_params['respawn']:
                self.poisons.append(Edible(**self.poison_params))

            return False

        def begin_goal_collision(arbiter, space, *args, **kwargs):

            # This is the goal, we end the simulation and update the measurements
            self.agent.update_meas('goal', 1)
            self.agent.reward += 100

            return False



        fruit_collision_handler = self.space.add_collision_handler(
            collision_type_a=0,
            collision_type_b=2
        )
        fruit_collision_handler.begin = begin_fruit_collision

        poison_collision_handler = self.space.add_collision_handler(
            collision_type_a=0,
            collision_type_b=3
        )
        poison_collision_handler.begin = begin_poison_collision

        goal_collision_handler = self.space.add_collision_handler(
            collision_type_a=0,
            collision_type_b=4
        )
        goal_collision_handler.begin = begin_goal_collision
        
    def step(self, action, disp):
        """
        Method called to execute an action in the environment.
        :param action: string, the string code for the action to be executed by the agent
        :return: a tuple containing :
            - sensory_input : the sensory input at time t+1
            - reward: the reward at time t
            - done: whether the episode is over
            - measurements : the measurements at time t+1
        """

        self.t += 1

        # Execute the action changes on the agent
        self.agent.apply_action(action)

        # Apply the step in the pymunk simulator
        self.reload_screen(disp)

        # Get the agent's position and orientation
        x, y = self.agent.body.position
        theta = self.agent.body.angle
        head_angle = self.agent.head
        self.agent.set_meas('x', x)
        self.agent.set_meas('y', y)
        self.agent.set_meas('theta', theta)
        self.agent.set_meas('head', head_angle)

        # Get the agent's perception
        for sensor in self.agent.sensors:
            sensor.get_sensory_input(self, disp)

        # Look for termination conditions
        if self.mode == 'goal' and self.agent.meas['goal'] == 1:
            self.done = True
        if self.mode == 'survival' and self.agent.meas['dead'] == 1:
            self.done = True
        if self.t >= self.horizon - 1:
            self.done = True
            
        return self.agent.state, self.agent.get_reward(), self.done, self.agent.get_meas()

    def reset(self):
        self.parameters['agent'].update(self.agent.get_new_averages())
        for sensor in self.agent.sensors:
            sensor.reset()
        self.__init__(**self.parameters)

    def drawEnvironment(self):
        for obstacle in self.obstacles:
            obstacle.draw()
