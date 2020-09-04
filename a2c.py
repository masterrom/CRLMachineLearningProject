import argparse
import math

import gym
import numpy as np
from itertools import count
from collections import namedtuple
from environment.environment import Robot, Environment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
import wandb
import time


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()



robot = Robot()
robot.newSection()
robot.newSection()
env = Environment(robot)

torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
wandb.init(project="Continuum A2C", name="A2C - (1000 Steps)- Xavier INIT Normalized")


def stableSoftMax(x):
    """
    stableSoftMax computes a normalized softmax
    :param x: Tensor List
    :return: Tensor List
    """
    x = torch.exp(x - torch.max(x))
    return x/torch.sum(x)


class Normalizer():

    def __init__(self, num_inputs):
        """
        Normalizer class is used to normalize a given state of the environment.
        This is done through determining a normal distribution of a given state
        :param num_inputs: int
        """
        self.n = torch.zeros(num_inputs)
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)

    def observe(self, x):
        """
        observe takes in a list, computes and stores the mean of the input
        :param x: Tensor list
        :return: None
        """
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        """
        normalize is specific to the state of length 6, where the last 2 elements
        are the coordinates of the goal point. The function normalized the first
        4 elements based on their mean, and the coordinates are normalized based on
        a 300x300 Box
        :param inputs: State of length 6
        :return: Normalized Tensor of the input
        """
        configs = inputs[0:4]
        point = inputs[4:]
        self.observe(configs)
        obs_std = torch.sqrt(self.var)
        normConfigs = (configs - self.mean)/obs_std

        normPoints = [point[0]/300, point[1]/300]

        return torch.cat((normConfigs, torch.tensor(normPoints)))



class Policy(nn.Module):
    """
    Policy class implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 8)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        #Xavier Initialization
        torch.nn.init.xavier_normal_(self.affine1.weight)
        torch.nn.init.xavier_normal_(self.action_head.weight)
        torch.nn.init.xavier_normal_(self.value_head.weight)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []


    def forward(self, input):
        """
        forward of both actor and critic
        :return: Output of both networks
        """

        x = F.relu(self.affine1(input))
        y = x.clone()

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = stableSoftMax(self.action_head(x))
        # action_prob = F.softmax(self.action_head(x), dim=-1)

        if torch.sum(action_prob.isnan()):
            print(y)
            print(action_prob)
            print(input)
            import ipdb; ipdb.set_trace()

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()

norm = Normalizer(4)


def select_action(state):
    """
    select_action picks a action based on the probability distribution of the output
    of the actor network
    :param state: envrionment Tuple
    :return: int
    """
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # probs[probs != probs] = 0.0
    if torch.sum(probs.isnan())>= 1:
        # import ipdb; ipdb.set_trace()
        print("issue ", torch.sum(probs.isnan()))

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)


    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # import ipdb;
    # ipdb.set_trace()

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    finish_episode is the training section of the network. Is called when a
    episode is finished. Training code. Calculates actor and critic loss and
    performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()


    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

    return loss


def main():
    """
    main function will start the training episode, where within each episode
    the robot can take atmost 1000 steps
    :return: None
    """

    args = parser.parse_args()


    running_reward = 10

    # run inifinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        env.reset()
        state = env.observation

        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 1000):


            # select action from policy
            action = select_action(np.array(state.state))
            action = env.robot.actions[action]

            # take the action
            # state, reward, done, _ = env.step(action)
            obs = env.robotStep(action[0], action[1])
            obs.state = norm.normalize(torch.tensor(obs.state))
            obs.nextState = norm.normalize(torch.tensor(obs.nextState))

            reward = obs.reward
            done = obs.done

            if args.render:
                env.render()

            model.rewards.append(reward)
            # wandb.log({"Actions": action, 'reward':reward}, step=t)
            ep_reward += reward
            if done:
                break

        print("End of Episode ", i_episode)
        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        loss = finish_episode()
        wandb.log({"Average Reward": running_reward, "loss": loss.item()}, step=i_episode)

        # torch.save(model.state_dict(), f"/u/meharabd/research/CRLMachineLearningProject/Models/{i_episode}.pth")
        torch.save(model.state_dict(),
                   f"/Users/master/Documents/School/Research/Snake/CRLMachineLearningProject/Models/episode-{i_episode}.pth")

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break



def test(path):
    """
    test function is used to upload a saved model, and test out the results
    :param path: System string path
    :return: None
    """
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    env.reset()
    state = env.observation

    env.render()

    while True:
        action = select_action(np.array(state.state))
        action = env.robot.actions[action]

        env.robotStep(action[0], action[1])
        env.render()



if __name__ == '__main__':

    # env.reset()
    # state = env.observation
    # norm = Normalizer(4)
    # import ipdb; ipdb.set_trace()
    main()
    # test("ModelPlaying/A2C 1/Models/episode-210.pth")
    # test("Models/episode-65.pth")