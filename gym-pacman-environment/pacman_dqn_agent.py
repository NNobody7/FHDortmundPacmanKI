import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import sleep

# env = gym.make("CartPole-v1")


# TODO set the desired number of games to play
episode_count = 500
score_threshold = 0.19

# Set to False to disable that information about the current state of the game are printed out on the console
# Be aware that the gameworld is printed transposed to the console, to avoid mapping the coordinates and actions
printState = True

# TODO Set this to the desired level
level = [
    ["#", "#", "#", "#", "#", "#", "#"],
    ["#", "*", "*", "P", "*", "*", "#"],
    ["*", "*", "#", "*", "#", "*", "*"],
    ["#", "*", "*", "*", "*", "*", "#"],
    ["#", "#", "*", "#", "*", "#", "#"],
    ["#", "H", "*", "*", "*", "R", "#"],
    ["#", "#", "#", "*", "#", "#", "#"],
]

# You can set this to False to change the agent's observation to Box from OpenAIGym - see also PacmanEnv.py
# Otherwise a 2D array of tileTypes will be used
usingSimpleObservations = False

# Defines all possible types of tiles in the game and how they are printed out on the console
# Should not be changed unless you want to change the rules of the game
tileTypes = {
    "empty": " ",
    "wall": "#",
    "dot": "*",
    "pacman": "P",
    "ghost_rnd": "R",
    "ghost_hunter": "H",
}

# Will be automatically set to True by the PacmanAgent if it is used and should not be set manually
usingPythonAgent = False


class PacmanAgent(gym.Wrapper):
    # Set the class attribute
    global usingPythonAgent
    usingPythonAgent = True

    def __init__(self, env_name="gym_pacman_environment:pacman-python-v0"):
        """ """
        super(PacmanAgent, self).__init__(gym.make(env_name))
        self.env_name = env_name
        self.action_space = self.env.action_space

    def act(self, action: int) -> str:
        """
        Convert the action from an integer to a string
        :param action: The action to be executed as an integer
        :return: The action to be executed as a string
        """
        match action:
            case 0:
                action = "GO_NORTH"
            case 1:
                action = "GO_WEST"
            case 2:
                action = "GO_EAST"
            case 3:
                action = "GO_SOUTH"
            case _:
                raise ValueError(f"Invalid action: {action}")
        return action

    def step(self, action: int) -> tuple:
        """
        Execute one time step within the environment
        :param action: The action to be executed
        :return: observation, reward, done, info
        """
        return self.env.step(self.act(action=action))

    def reset(self) -> object:
        """
        Resets the state of the environment and returns an initial observation.
        :return: observation (object): the initial observation of the space.
        """
        return self.env.reset()

# Select which gym-environment to run
env = PacmanAgent()

# print("action space", env.action_space.n)
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
# a = env.reset()
# print(len(a))
# state, info = env.reset()
state = env.reset()
n_observations = state.size

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # policy_net(state).max(1).indices.view(1, 1)
            # print('policy net state', policy_net(state).max(1).indices.max(1).indices.view(1,1), '\n')
            return policy_net(state).max(1).indices.view(1, 1)
            # return policy_net(state).max(1).indices.max(1).indices.view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    # action_batch = torch.cat(batch.action).view(-1, 1)
    reward_batch = torch.cat(batch.reward)
    # print('dimension of action batch', action_batch.size())
    # print('dimension of state batch', state_batch.size())

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(action_batch)
    # print(state_batch)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # use state_batch 
    '''
    write state action values (variable name) array brackets
    state_action_values[variable]
    {
    
    }
    [
    
    ]
    '''

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.backends.mps.is_available():
    num_episodes = 500 #600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()
    # print('original state size: ', torch.tensor(state, dtype=torch.float32).shape)
    state = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).flatten().unsqueeze(0)

        # print('state size: ', state.shape)
        # print('action size: ', action.shape)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()



# Save the trained model
torch.save(policy_net.state_dict(), 'pacman_dqn_agent01.pth')
print('saved the model')

print('playing')
# Set the policy_net in evaluation mode
policy_net.eval()

# Number of episodes to play for testing
num_test_episodes = 10
env = PacmanAgent()
for i_episode in range(num_test_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)


    for t in count():
        # Use the trained policy network to select actions
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)

        # Take the action in the environment
        observation, reward, terminated, truncated = env.step(action.item())

        # Display the environment
        env.render()
        # print('rendering')

        # Update the state for the next step
        state = torch.tensor(observation, dtype=torch.float32, device=device).flatten().unsqueeze(0)
        sleep(0.1)

        # Check if the episode is done
        if terminated or truncated:
            break
    
    print('playing next match')
# Close the environment after testing
env.close()
