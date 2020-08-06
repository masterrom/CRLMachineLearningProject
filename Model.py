import math
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import random, sample
from environment.environment import section, Environment, Observation, Robot
import turtle
import wandb
from tqdm import tqdm




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
            nn.Linear(observationShape, 32),
            nn.ReLU(),
            nn.Linear(32, numActions),
            nn.ReLU()
        )

        self.opt = optim.Adam(self.net.parameters(),lr=0.0001,)

    def forward(self, x):
        return self.net(x)


def trainStep(stateTransitions, model, targetModel, numActions):
    currentState = torch.stack([torch.Tensor(s.state) for s in stateTransitions])
    rewards = torch.stack([torch.Tensor([s.reward]) for s in stateTransitions])
    nextState = torch.stack([torch.Tensor(s.nextState) for s in stateTransitions])
    # mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in stateTransitions])
    actions = [s.action for s in stateTransitions]

    with torch.no_grad():
        qValNext = targetModel(nextState).max(-1)[0] # should output (N, numActions)

    model.opt.zero_grad()
    qVal = model(currentState)
    oneHotActions = F.one_hot(torch.LongTensor(actions), numActions)
    gamma = 0.01

    loss = ((rewards + gamma * qValNext - torch.sum(qVal * oneHotActions, -1)) ** 2).mean()

    loss.backward()
    model.opt.step()

    # import ipdb;
    # ipdb.set_trace()

    return loss

# Copying over the weights from m to tgt
def updateTGTModel(m, tgt):
    tgt.load_state_dict(m.state_dict())


def main(test=False, chkpt=None):

    if not test:
        wandb.init(project="MultiSection Continum", name="Reaching Task 32 Per Layer")

    robot = Robot()
    robot.newSection()
    robot.newSection()

    env = Environment(robot)
    if test:
        env.staticPoint([-75, 150])
        env.render()
    else:
        env.staticPoint([-75, 150])

    lastObs = env.getObservation()


    rb = ReplayBuffer()

    minRBSize = 10000
    sampleSize = 2500
    envStepsBeforeTrain = 100
    targetModelUpdate = 150

    epsMin = 0.01
    epsDecay = 0.99998


    model = Model(len(lastObs.state), len(env.robot.actions))
    if chkpt != None:
        model.load_state_dict(torch.load(chkpt))

    targetModel = Model(len(lastObs.state), len(env.robot.actions))
    updateTGTModel(model, targetModel)

    stepSinceTrain = 0
    stepSinceTGTUpdate = 0
    stepNum = -1 * minRBSize

    episodeRewards = []
    rollingReward = 0

    # Copying over the weights
    tq = tqdm()
    # Work in progress
    while True:
        if test:
            env.render()
            time.sleep(0.05)
        tq.update(1)
        eps = epsDecay ** (stepNum/100)
        if test:
            eps = 0


        if random() < eps:
            action = env.robot.randomAction()
        else:
            actNum = model(torch.tensor(lastObs.state)).max(-1)[-1].item()
            action = env.robot.actions[actNum]

        obs  = env.robotStep(action[0], action[1])
        rollingReward = obs.reward

        # print(obs)
        # # env.render()
        # x = model(torch.Tensor(obs.state))
        # # print(x)
        #
        episodeRewards.append(rollingReward)
        #
        # if stepSinceTGTUpdate > targetModelUpdate:
        # # if env.done():
        #     episodeRewards.append(rollingReward)
        #     if test:
        #         print(rollingReward)
        #     print(episodeRewards)
        #     rollingReward = 0
        #     # env.reset()

        obs.reward = obs.reward / 100

        stepSinceTrain += 1
        stepNum += 1
        rb.insert(obs)
        if (not test) and len(rb.buffer) >= minRBSize and stepSinceTrain > envStepsBeforeTrain:
            stepSinceTGTUpdate += 1
            loss = trainStep(rb.sample(sampleSize),model, targetModel, len(env.robot.actions))
            wandb.log({"Loss": loss.detach().item(), "eps": eps, "Step Rewards:": np.mean(episodeRewards)}, step=stepNum)
            stepSinceTrain = 0

            if stepSinceTGTUpdate > targetModelUpdate:
                print("Updating Target Model")
                updateTGTModel(model, targetModel)
                stepSinceTGTUpdate = 0
                torch.save(targetModel.state_dict(), f"Models/{stepNum}.pth")
                episodeRewards = []

            # print(stepNum, loss.detach().item())
            # import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    # main(True, "Models/2623071.pth")
    main()




