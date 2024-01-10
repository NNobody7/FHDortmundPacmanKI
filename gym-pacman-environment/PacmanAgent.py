import gym
from gym import logger
from time import sleep

# TODO set the desired number of games to play
episode_count = 500
score_threshold = 0.19

# Set to False to disable that information about the current state of the game are printed out on the console
# Be aware that the gameworld is printed transposed to the console, to avoid mapping the coordinates and actions
printState = True

# TODO Set this to the desired level
# level = [
#     ["#", "#", "#", "#", "#", "#", "#"],
#     ["#", "*", "*", "P", "*", "*", "#"],
#     ["*", "*", "#", "*", "#", "*", "*"],
#     ["#", "*", "*", "*", "*", "*", "#"],
#     ["#", "#", "*", "#", "*", "#", "#"],
#     ["#", "H", "*", "*", "*", "R", "#"],
#     ["#", "#", "#", "*", "#", "#", "#"],
# ]

level = []
with open('Pacman_Level\RL03_square_tunnel_R.pml') as levelFile:
    for line in levelFile:
        level.append([*line][:-1])


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

if __name__ == "__main__":
    # Can also be set to logger.WARN or logger.DEBUG to print out more information during the game
    logger.set_level(logger.DISABLED)

    # Select which gym-environment to run
    env = PacmanAgent()

    training_data = []
    accepted_scores = []

    # Execute all episodes by resetting the game and play it until it is over
    for i in range(episode_count):
        if(i % 10 == 0):
            print(i)
        game_memory = []
        prev_state = []
        observation = env.reset()
        reward = 0
        done = False
        while True:
            # Determine the agent's next action based on the current observation and reward and execute it
            env.render()
            # TODO better action selection
            action = env.action_space.sample()
            if(len(prev_state) > 0):
                game_memory.append([prev_state, action])
            observation, reward, done, debug = env.step(action)
            prev_state = observation
            if done:
                if reward >= score_threshold:
                    accepted_scores.append(reward)
                    for data in game_memory:
                        output = [0] * 4
                        output[data[1]] = 1
                        training_data.append([data[0], output])
                break
    print(accepted_scores)
    print(len(accepted_scores))
    env.close()
