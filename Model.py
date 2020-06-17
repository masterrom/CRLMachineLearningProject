import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from random import random, sample
from environment.environment import section, Environment, Observation
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

class Model(nn.Module):
    def __init__(self, observationShape, numActions):
        super(Model, self).__init__()
        self.observationShape = observationShape
        self.numActions = numActions

        self.net = nn.Sequential(
            nn.Linear(observationShape[0], 256),
            nn.ReLU(),
            nn.Linear(256, numActions)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':

    # turtle.setup(800, 600)
    # wn = turtle.Screen()
    # wn.tracer(300)

    arcLength = 100
    robot = section(arcLength, 120)
    base = Environment(robot)
    base.render()
    # model = Model(3, 4)
    # qVal = model(torch.Tensor(base.observation.nextState))
    # import ipdb; ipdb.set_trace()
    # base.drawGround()
    commands = ['l', 'r', 'e', 'c']

    for i in range(100):
        direction = base.randomAction()
        base.robotStep(commands[direction])
        base.render()
