import numpy as np
import gym
import random
import time
from IPython.display import clear_output
import sys

# Defining environment
env = gym.make("FrozenLake-v0")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

# Initializing q table
q_table = np.zeros((state_space_size, action_space_size))

# print(q_table)

# Settings of the simulation

num_episodes = 40000
max_steps_per_episode = 100

# Alpha
learning_rate = 0.01
# Gamma
discount_rate = 0.99

# Exploration/exploitation tradeoff
# Epsilon
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

holes = [5, 7, 11, 12]

#Traning
for episode in range(num_episodes):

    if episode % 1000 == 0:
        print("Processing episode ", str(episode))
        # print("############### EXPLORATION RATE :", exploration_rate)
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # We first determine wether to explore or exploit
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        # Execution of the action
        new_state, reward, done, info = env.step(action)

        if new_state in holes:
            reward = -400
        elif new_state == 15:
            reward = 500
        else:
            reward = -2

        # Update of Q-Table given the reward rewards_current_episode
        q_table[state, action] = q_table[state, action] * (1-learning_rate) + \
         learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # Transition to the next state
        state = new_state
        rewards_current_episode += reward

        if done == True:
             break

    # exploration_rate -= exploration_decay_rate
    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * \
        np.exp(-exploration_decay_rate * episode)


    rewards_all_episodes.append(rewards_current_episode)

# Once all episodes are done, calculate and print average rewards
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000

print("####### Average reward per thousand episodes #######")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
print("\n ####### Q-Table #######")
print(q_table)

# Playing
# sys.exit(1)
numbers_of_play = 1000
wins = 0
for episode in range(numbers_of_play):
    state = env.reset()
    done = False
    # print("\n\n####### Processing episode ", episode+1," #######\n\n")
    # time.sleep(1)
    for step in range(max_steps_per_episode):
        # clear_output(wait=True)
        # env.render()
        # time.sleep(0.3)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)
        if done == True:
            # clear_output(wait=True)
            # env.render()
            if new_state == 15:
                print("####### You reached the goal #######")
                wins += 1
                # time.sleep(3)
            else:
                print("You fell in a hole")
                # time.sleep(3)
            # clear_output()
            break
        state = new_state
env.close()
rate = wins / numbers_of_play
print(str(rate))
