import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from random import random, sample
from environment.environment import section, Environment, Observation, Robot
import turtle


class ReplayBuffer:

    def __init__(self, bufferSize=100000):
        self.bufferSize = bufferSize
        self.buffer = deque(maxlen=bufferSize)

    def insert(self, obs):
        self.buffer.append(obs)
        # self.buffer = self.buffer[-self.bufferSize:]

    def sample(self, numSample):
        assert numSample <= len(self.buffer)
        return sample(self.buffer, numSample)


# Copying over the weights from m to tgt
def updateTGTModel(m, tgt):
    tgt.load_state_dict(m.state_dict())


class Model(nn.Module):
    def __init__(self, observationShape, numActions):
        super(Model, self).__init__()
        self.observationShape = observationShape
        self.numActions = numActions

        self.net = nn.Sequential(
            nn.Linear(observationShape, 256),
            nn.ReLU(),
            nn.Linear(256, numActions)
        )

    def forward(self, x):
        return self.net(x)


# def trainStep(stateTransitions, model, targetModel)


if __name__ == '__main__':

    arcLength = 100

    robot = Robot()
    robot.addSection()
    robot.addSection()

    base = Environment(robot)
    base.render()

    turtle.Screen().update()

    robot.sections[-1].section.color('black')

    # model = Model(3, 4)
    # targetModel = Model(3, 4)
    # updateTGTModel(model, targetModel)
    x = turtle.Turtle()
    x.color('purple')
    x.width(5)
    while True:
        direction = str(input("Enter direction: "))

        for i in range(10):
            base.robotStep(1, direction)
            base.render()


    # commands = ['l','r','e','c']
    # while True:
    #     direction = int(input("Enter direction: "))
    #     numSteps = int(input("Enter steps in direction " + commands[direction]))
    #     for j in range(numSteps):
    #         base.robotStep(commands[direction])
    #         # base.render()
    #         print(base.robot.getTipPos())

