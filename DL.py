import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from collections import namedtuple
from itertools import count
import math
import matplotlib
import matplotlib.pyplot as plt
from math import exp
from taxi_env import TaxiEnv



#Selecting simulation
#environment="CartPole-v0"
environment = "TaxiEnv"
MAP_SIZE = 8
# environment="MsPacman-v0"

#Simulation hyperparameters
BATCH_SIZE = 10
GAMMA = 0.99 #Discount rate
MAX_EPSILON = 1 #Maximum exploration rate
MIN_EPSILON = 0.01 #Minimum exploration rate
EPSILON_DECAY = 0.995#Epsilon decay rate
MEMORY_SIZE = 1000
ALPHA = 0.001 #Learning rate
num_episodes = 1000

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward', 'done')
)
class DQN():
    def __init__(self, input, output):

        self.output = output
        self.memory = deque(maxlen = MEMORY_SIZE)
        self.exploration_rate = MAX_EPSILON

        self.model = Sequential()
        self.model.add(Dense(8, input_shape = (input,), activation="relu"))
        self.model.add(Dense(12, activation="relu"))
        self.model.add(Dense(self.output, activation="softmax"))
        self.model.compile(loss="mse", optimizer=Adam(lr = ALPHA))



    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, next_state, reward, done))

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.output)
        else:
            q_values = self.model.predict(state)
        #    print(q_values)
            return np.argmax(q_values[0])

    def replay_memory(self, episode):
        if len(self.memory) < BATCH_SIZE:
            return
        sample = random.sample(self.memory, BATCH_SIZE)
        for state, action, next_state, reward, done in sample:
            q_update = reward
            if done == False:
                q_update = (reward + GAMMA* np.amax(self.model.predict(next_state)))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate = self.exploration_rate * EPSILON_DECAY
        self.exploration_rate = max(MIN_EPSILON, self.exploration_rate)
        #if episode > 150:
        #    self.exploration_rate = 0


def plot(values, moving_avg_period):
    mvg = []
    plt.figure(2)
    plt.clf()
    plt.title("Training")
    plt.xlabel('Episode')
    plt.ylabel('Steps to complete')
    plt.plot(values)
    mov = get_moving_average(moving_avg_period, values)
    print("Average:", mov)
    mvg.append(mov)
    plt.plot(mvg)
    plt.pause(0.001)

    # print("Episode", len(values), "\n", \
    #     moving_avg_period, "episode moving average:", moving_average[], "with epsilon", strategy.start )
#    display.clear_output(wait=True)


def get_moving_average(period, values):
    if len(values) >= period:
        mvg = (sum(values[-period:-1])/period)
        return mvg
    else:
        return 0

def main():
    episode_durations = []
    wins = 0

    #env = gym.make(environment)
    env = TaxiEnv(MAP_SIZE)
    # observation_space = env.observation_space.shape[0]
    # action_space = env.action_space.n
    observation_space = 1
    action_space = 4
    dqn = DQN(observation_space, action_space)

    for i in range(num_episodes):
        done = False
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        reward_memory = 0
        steps = 0
        if (i % 5 == 0):
            print("Processed ", i/num_episodes * 100)
        while not done and steps < 20:
            steps +=1
            #env.render()
            action = dqn.select_action(state)
        #    print(action)
            next_state, reward, done, _ = env.step(action)
            # if done == True:
            #     reward = -10
            # if reward == 1:
            #    reward = reward * (10 - exp(abs(state[0][2])*5))
            reward_memory += reward
            #print(reward)
            next_state = np.reshape(next_state, [1, observation_space])
            dqn.add_to_memory(state, action, reward, next_state, done)
            state = next_state
            dqn.replay_memory(i)
            if done:
               wins += 1
               break
    print(wins)
    print(sum(episode_durations) / num_episodes)

if __name__ == "__main__":
    main()
