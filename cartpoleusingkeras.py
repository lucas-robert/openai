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

#Selecting simulation
environment="CartPole-v0"

#Simulation hyperparameters
BATCH_SIZE = 20
GAMMA = 0.95 #Discount rate
MAX_EPSILON = 1 #Maximum exploration rate
MIN_EPSILON = 0.01 #Minimum exploration rate
EPSILON_DECAY = 0.995 #Epsilon decay rate
MEMORY_SIZE = 1000000
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
        self.model.add(Dense(24, input_shape = (4,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.output, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr = ALPHA))

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, next_state, reward, done))

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.output)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def replay_memory(self):
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

def plot(values, moving_avg_period):
    mvg = []
    plt.figure(2)
    plt.clf()
    plt.title("Training")
    plt.xlabel('Episode')
    plt.ylabel('Duration')
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

def CartPole():
    episode_durations = []
    env = gym.make(environment)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn = DQN(observation_space, action_space)

    for i in range(num_episodes):
        done = False
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        for timestep in count():
            env.render()
            action = dqn.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if done == True:
                reward = -10
            if reward == 1:
               reward = reward * (10 - exp(abs(state[0][2])*5))
            next_state = np.reshape(next_state, [1, observation_space])
            dqn.add_to_memory(state, action, reward, next_state, done)
            state = next_state
            dqn.replay_memory()
            if done:
                print("Episode ", i, "done with ", timestep)
                episode_durations.append(timestep)
                plot(episode_durations, 100)
                break

if __name__ == "__main__":
    CartPole()
