import numpy as np
import gym
from model import Agent
import wandb
from gym import wrappers
import wandb

print("Begining Main")

if __name__ == '__main__':
    print("Begining Main")
    agent = Agent(alpha=0.00001, beta=0.0005, inputDim=4, gamma=0.99,
                  nActions=2, l1Size=32, l2Size=32)

    wandb.init(project="Cartpole A2C", name="Actor critic Method")

    env = gym.make('CartPole-v0')
    scoreHistory = []
    nEpisodes = 2500
    i = 0
    while True:
        done = False
        score = 0
        prevObs = env.reset()
        scoreHistory = []
        while not done:
            action = agent.chooseAction(prevObs)
            obs, reward, done, info = env.step(action)
            score += reward
            agent.learn(prevObs,reward, obs, done)
            prevObs = obs

            scoreHistory.append(reward)


        # print("episode ", i , " Score ", score)
        wandb.log({"Eps Avg reward:": np.mean(scoreHistory), "Eps cumulative Reward": score},
                  step=i)
        if score >= 200:
            print("Goal reward achieved")
            break
        i += 1
        # scoreHistory.append(score)

    # print(scoreHistory)