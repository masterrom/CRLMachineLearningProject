import torch.distributions as Distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np

# 2 different networks, actor: Outputs a action given a input state, critic: takes the action produced by the actor and criticizes it

class GenericNetwork(nn.Module):

    def __init__(self, lr, inputDim, fc1Dim, fc2Dim, nActions):
        super(GenericNetwork, self).__init__()

        self.lr = lr
        self.inputDim = inputDim
        self.fc1Dim = fc1Dim
        self.fc2Dim = fc2Dim
        self.nActions = nActions

        self.network = nn.Sequential(
            nn.Linear(self.inputDim,self.fc1Dim),
            nn.ReLU(),
            nn.Linear(self.fc1Dim, self.fc2Dim),
            nn.ReLU(),
            nn.Linear(self.fc2Dim, self.nActions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        return self.network(state)




class Agent(object):

    def __init__(self, alpha, beta, inputDim, gamma=0.99, l1Size=256, l2Size=256, nActions=2):

        self.gamma = gamma
        self.logProbs = None
        '''
        LogProbs: Actor critic works by updating the actor network from the gradient of the log of the policy
        Policy is simply a probability distribution
        '''

        self.actor = GenericNetwork(alpha, inputDim, l1Size, l2Size, nActions)
        self.critic = GenericNetwork(beta, inputDim, l1Size, l2Size, nActions=1)


    def chooseAction(self, obs):

        probabilities = F.softmax(self.actor.forward(obs)) # So all action probabilities add upto 1
        actionProbs = Distributions.Categorical(probabilities)
        action = actionProbs.sample()

        self.logProbs = actionProbs.log_prob(action)

        return action.item()

    def learn(self, state, reward, newState, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        #Value of current state
        criticValue = self.critic.forward(state)
        newStateCriticVal = self.critic.forward(newState)

        delta = ((reward + self.gamma*criticValue*(1- int(done))) - newStateCriticVal)

        actorLoss = self.logProbs * delta
        criticLoss = delta ** 2

        (actorLoss + criticLoss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()























