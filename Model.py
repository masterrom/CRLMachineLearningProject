import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Model(nn.Module):
    def __init__(self, observationShape, numActions):
        super(Model, self).__init__()
        self.observationShape = observationShape
        self.numActions = numActions

        self.net = nn.Sequential(
            nn.Linear(observationShape, 256),
            nn.ReLU(),
            nn.Linear(256, numActions),
            nn.ReLU()
        )

        self.opt = optim.Adam(self.net.parameters(),lr=0.0001,)

    def forward(self, x):
        return self.net(x)


def trainStep(stateTransitions, model, targetModel, numActions):
    currentState = torch.stack([torch.Tensor(s.state) for s in stateTransitions])
    rewards = torch.stack([torch.Tensor([s.reward]) for s in stateTransitions])
    nextState = torch.stack([torch.Tensor(s.nextState) for s in stateTransitions])
    mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in stateTransitions])
    actions = [s.action for s in stateTransitions]

    with torch.no_grad():
        qValNext = targetModel(nextState).max(-1)[0] # should output (N, numActions)

    model.opt.zero_grad()
    qVal = model(currentState)
    oneHotActions = F.one_hot(torch.LongTensor(actions), numActions)

    loss = ((rewards + mask[:, 0] * qValNext - torch.sum(qVal * oneHotActions, -1)) ** 2).mean()
    loss.backward()
    model.opt.step()

    return loss

# Copying over the weights from m to tgt
def updateTGTModel(m, tgt):
    tgt.load_state_dict(m.state_dict())


if __name__ == '__main__':

    robot = Robot()
    robot.newSection()
    robot.newSection()

    env = Environment(robot)
    obs = env.getObservation()

    # env.render()
    # turtle.Screen().update()
    rb = ReplayBuffer()


    model = Model(len(obs.state), len(env.robot.actions))
    targetModel = Model(len(obs.state), len(env.robot.actions))

    # Copying over the weights
    updateTGTModel(model, targetModel)



    while True:
        action = env.robot.randomAction()
        obs  = env.robotStep(action[0], action[1])
        # print(obs)
        # env.render()
        x = model(torch.Tensor(obs.state))
        # print(x)
        rb.insert(obs)
        if len(rb.buffer) >= 1000:
            loss = trainStep(rb.sample(250),model, targetModel, len(env.robot.actions))
            print(loss)
            import ipdb; ipdb.set_trace()


        if env.done():
            env.reset()

    # model = Model()

    # NOTE:Demo
    '''
    while True:
        secNum = int(input("Enter SecNum: "))
        direction = str(input("Enter direction: "))
        steps = int(input("Enter number of steps: "))

        for i in range(steps):
            env.robotStep(secNum, direction)
            env.render()
    '''






