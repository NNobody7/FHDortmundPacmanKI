import PacmanAgent
import torch
from itertools import count
from time import sleep
from models import *


num_test_episodes = 10

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


levelName = "RL02_square_tunnel_H.pml"
PacmanAgent.setLevel(levelName)

env = PacmanAgent.PacmanAgent()

n_actions = env.action_space.n
state = env.reset()
n_observations = state.size
policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load(levelName+".pth"))

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
