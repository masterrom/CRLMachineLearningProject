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
        """
        ReplayBuffer is to used to keep a list of timeSteps that the agent has taken in the environment
        :param bufferSize: Int
        """
        self.bufferSize = bufferSize
        self.buffer = [None]*bufferSize
        self.index = 0

    def insert(self, obs):
        """
        insert is used to add an observation to the buffer
        :param obs: Observation item
        :return: None
        """
        self.buffer[self.index % self.bufferSize] = obs
        self.index += 1

    def sample(self, numSample):
        """
        sample is used to pick numSamples from the buffer
        :param numSample: int
        :return: list of Observation Items
        """
        # assert numSample <= len(self.buffer)
        assert numSample < min(self.index, self.bufferSize)
        # if numSample > min(self.index, self.bufferSize)

        if self.index < self.bufferSize:
            return sample(self.buffer[:self.index], numSample)
        return sample(self.buffer, numSample)


class Model(nn.Module):
    def __init__(self, observationShape, numActions):
        """
        Model is a generic class, generates a Neural Network that takes in input of size state, and outputs the
        number of actions
        :param observationShape: length of the state (int)
        :param numActions: number of actions (int)
        """
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
        """
        Forward Pass of the network
        :param x: state
        :return: output of neural net
        """
        return self.net(x)


def trainStep(stateTransitions, model, targetModel, numActions, device):
    """
    trainStep computes the loss and preforms backpropagation
    :param stateTransitions: list of states
    :param model: Network 1 (Model)
    :param targetModel: Network 2 (Model)
    :param numActions: Number of actions (int)
    :param device: (string)
    :return: Loss (float)
    """
    currentState = torch.stack([torch.Tensor(s.state) for s in stateTransitions]).to(device)
    rewards = torch.stack([torch.Tensor([s.reward]) for s in stateTransitions]).to(device)
    nextState = torch.stack([torch.Tensor(s.nextState) for s in stateTransitions]).to(device)
    # mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in stateTransitions])
    actions = [s.action for s in stateTransitions]

    with torch.no_grad():
        qValNext = targetModel(nextState).max(-1)[0] # should output (N, numActions)

    model.opt.zero_grad()
    qVal = model(currentState)
    oneHotActions = F.one_hot(torch.LongTensor(actions), numActions).to(device)
    gamma = 1

    loss = ((rewards + gamma * qValNext - torch.sum(qVal * oneHotActions, -1)) ** 2).mean()

    loss.backward()
    model.opt.step()

    # import ipdb;
    # ipdb.set_trace()

    return loss

def updateTGTModel(m, tgt):
    """
    updateTGTModel copies the weight of m into tgt
    :param m: Network 1
    :param tgt: Network 2
    :return: None
    """
    tgt.load_state_dict(m.state_dict())


def main(test=False, chkpt=None, device='cuda'):
    """
    main is used to start and preform the training in non-render mode
    :param test: Not required
    :param chkpt: Not required
    :param device: string (cuda or cpu)
    :return: None
    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not test:
        wandb.init(project="MultiSection Continum", name="Reaching Task 32 Per Layer")

    robot = Robot()
    robot.newSection()
    robot.newSection()

    env = Environment(robot)
    if test:
        # env.staticPoint([-9.966711079379195, 99.3346653975306])
        env.render()
    # else:
    #     env.staticPoint([-9.966711079379195, 99.3346653975306])

    lastObs = env.getObservation()


    rb = ReplayBuffer()

    memorySize = 500000
    minRBSize = 20000

    sampleSize = 750

    envStepsBeforeTrain = 250

    targetModelUpdate = 500


    epsMin = 0.01
    epsDecay = 0.99999


    model = Model(len(lastObs.state), len(env.robot.actions)).to(device)
    if chkpt != None:
        model.load_state_dict(torch.load(chkpt))

    targetModel = Model(len(lastObs.state), len(env.robot.actions)).to(device)
    updateTGTModel(model, targetModel)

    stepSinceTrain = 0
    # stepSinceTrain keeps track of the number of steps since the last main network training
    # in this case main network updates after every envStepsBeforeTrain

    stepSinceTGTUpdate = 0
    # stepSinceTGTUpdate keeps track of the number of steps since the last target network update (ie transfering main network weights)
    # in this case the target network updates after every targetModelUpdate

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
        eps = epsDecay ** (stepNum/10)
        if test:
            eps = 0


        if random() < eps:
            # print("Taking random action")
            action = env.robot.randomAction()
        else:
            actNum = model(torch.tensor(lastObs.state).to(device)).max(-1)[-1].item()
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
        if env.done():
            env.reset()
            # env.staticPoint([-9.966711079379195, 99.3346653975306])

        # obs.reward = obs.reward / 100

        stepSinceTrain += 1
        stepNum += 1
        rb.insert(obs)
        if (not test) and rb.index >= minRBSize and stepSinceTrain > envStepsBeforeTrain:
            stepSinceTGTUpdate += 1
            loss = trainStep(rb.sample(sampleSize),model, targetModel, len(env.robot.actions), device)
            wandb.log({"Loss": loss.detach().cpu().item(), "eps": eps, "Step Rewards:": np.mean(episodeRewards)}, step=stepNum)
            stepSinceTrain = 0

            if stepSinceTGTUpdate > targetModelUpdate:
                print("Updating Target Model")
                updateTGTModel(model, targetModel)
                stepSinceTGTUpdate = 0
                torch.save(targetModel.state_dict(), f"/u/meharabd/research/CRLMachineLearningProject/Models/{stepNum}.pth")
                episodeRewards = []

            # print(stepNum, loss.detach().item())
            # import ipdb; ipdb.set_trace()


def modelTest(test=False, chkpt=None, device='cuda'):
    """
    modelTest is used to upload a saved model, and test out the results
    :param test: Bool should be set to true
    :param chkpt: string path of where the model exists
    :param device: cuda or cpu
    :return: None
    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        model.load_state_dict(torch.load(chkpt, map_location=torch.device('cpu')))

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
            loss = trainStep(rb.sample(sampleSize),model, targetModel, len(env.robot.actions), device)
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
    modelTest(True, "ModelPlaying/Test 4/Models/5566009.pth")
    # main()




