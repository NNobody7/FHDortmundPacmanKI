import gym
from gym import logger
from time import sleep
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

from DQN import DQN
from ReplayMemory import Transition

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO set the desired number of games to play
if torch.cuda.is_available():
    episode_count = 200
else:
    episode_count = 50
score_threshold = 0.19

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

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
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
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
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

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
    
if __name__ == "__main__":
    # Can also be set to logger.WARN or logger.DEBUG to print out more information during the game
    logger.set_level(logger.DISABLED)

    # Select which gym-environment to run
    env = PacmanAgent()

    
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    training_data = []
    accepted_scores = []
    episode_durations = []
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    # Execute all episodes by resetting the game and play it until it is over
    for i in range(episode_count):
        if(i % 10 == 0):
            print(i)
        game_memory = []
        prev_state = []
        observation = env.reset()
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        # print(state)
        reward = 0
        done = False
        for t in count():
            # Determine the agent's next action based on the current observation and reward and execute it
            env.render()
            action = select_action(state)
            print(action)
            observation, reward, done, debug = env.step(action.item())
            prev_state = observation
            if(len(prev_state) > 0):
                game_memory.append([prev_state, action, observation, reward])
            optimize_model(game_memory)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t+1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

    env.close()
