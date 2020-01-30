import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from IPython import display

#Selecting simulation
environment="CartPole-v0"

#Simulation hyperparameters
batch_size = 100
gamma = 0.99 #Discount rate
eps_start = 1 #Maximum exploration rate
eps_end = 0.01 #Minimum exploration rate
eps_decay = 0.997 #Epsilon decay rate
target_update = 1 #Target network is updated every 10 episodes
memory_size = 100000
lr = 0.001 #Learning rate
num_episodes = 1000

#Class that holds the implmentation of QDN
class DQN(nn.Module):
    def __init__(self, img_height,img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24) #IN : Number of pixels * number of color channel
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        # self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, t): #Required for all Pytorch deep network
        print("xxxxxxxx", t.shape)
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        # t = F.relu(self.fc3(t))
        t = self.out(t)
        return t

# Class that holds experience tuple
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # Act as a queue
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

# Class that holds Epsilon value actualisation (Exploration/exploitation trade-off)
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        # max_exploration_rate
        self.start = start
        # min_exploration_rate
        self.end = end
        #decay_rate
        self.decay = decay

# Calculating exponential decay rate
    def get_exploration_rate(self, current_step):
        self.start *= self.decay
        return max(self.start, self.end)

class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        # Instance of EpsilonGreedyStrategy class
        self.strategy = strategy
        # number of action the agent can take
        self.num_actions = num_actions
        # Device used by torch(GPU/CPU)
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions) #Exploration
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad(): #Disable gradient tracking
                return policy_net(state).argmax(dim=1).to(self.device) #Exploitation using the policy neural network. Chosing highest Q-value output


# Class that manages the environment
class EnvManager():
    def __init__(self, device, environment):
        self.device = device
        self.env = gym.make(environment).unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self): #Reset environment and set current screen to None (not rendered yet)
        self.env.reset()
        self.current_screen = None

    def close(self): #Close environment
        self.env.close()

    def render(self, mode='human'): #Render environment
        return(self.env.render(mode))

    def num_actions_available(self): #Get amount of possible action
        return self.env.action_space.n

    def take_action(self, action): #Send action to DQN
        _, reward, self.done, _ = self.env.step(action.item()) #using .item() on tensor to get the value
        if reward > 0:
            reward = 1
        else:
            reward = -1
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self): #get processed image of the screen. A state is the difference between current and previous frame
        if self.just_starting() or self.done: #If we're starting or if episode is done, we render a black screen
            self.current_screen = self.get_processed_screen() #New black Screen of the same shape of current screen
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2,0,1)) #????????????
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:,top:bottom, :] #Strip the top 40% and bottom 20% of the image
        return screen

    def transform_screen_data(self, screen):
        #Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255 #getting individual pixel value
        screen = torch.from_numpy(screen)

        #Torchvision functions to compose image transforms
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((40,90)),
            T.ToTensor()
        ])
        return resize(screen).unsqueeze(0).to(self.device) #add a batch dimesion


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title("Training")
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    moving_average = get_moving_average(moving_avg_period, values)
    plt.plot(moving_average)
    plt.pause(0.001)

    print("Episode", len(values), "\n", \
        moving_avg_period, "episode moving average:", moving_average[-1])
    display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period: #Return 0, as it's impossible to calculate
        moving_avg = values.unfold(dimension=0, size=period, step=1)\
        .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period -1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1)) #Predicted Q values

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1)\
        .max(dim=1)[0].eq(0).type(torch.bool) #Looking for final state. We flatten the next_states and look for a Q-value of 0, which is when the episode stops
        #We put a value of True to final states, and a value of False to the non ending states

        non_final_state_locations = (final_state_locations == False) #select all next_state that don't end the game
        non_final_states = next_states[non_final_state_locations] #get all corresponding non final states
        batch_size = next_states.shape[0] #How many next_state are not ending the game
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach() #We fill the tensor with Q values for each states that are not ending
        return values

def extract_tensors(experiences):
    batch = Experience(*zip(*experiences)) #Extract array of experiences and put it in a new instance

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return(t1,t2,t3,t4)

#Simulation setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = EnvManager(device, environment)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

#DQN setup
policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device) #Create first DQN
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)#Create second DQN
target_net.load_state_dict(policy_net.state_dict()) #Copy model from policy to target
target_net.eval() #Not in Training mode
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states) #Best Q Values in the future
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break

    if episode % target_update == 0: #Update of the target net
        target_net.load_state_dict(policy_net.state_dict())
em.close()
