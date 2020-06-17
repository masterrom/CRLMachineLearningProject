import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from random import random, sample


class ReplayBuffer:
    def __init__(self, bufferSize=100000):
        super().__init__()
        self.bufferSize = bufferSize
        self.buffer = []

    def insert(self, sars):
        self.buffer.append(sars)
        self.buffer = self.buffer[-self.bufferSize:]

    def sample(self, numSamples):
        assert numSamples <= len(self.buffer)
        return sample(self.buffer, numSamples)



if __name__ == '__main__':
