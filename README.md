# Pacman Implementation with OpenAI Gym Support
This is a simple implementation of Pacman game with [OpenAI Gym](https://www.gymlibrary.dev) support. The game is implemented in Python 3.10 and uses [PyGame](https://www.pygame.org/news) for rendering the game. The game is implemented as an OpenAI Gym environment and can be used with any reinforcement learning algorithm that supports OpenAI Gym environments.

## <span style="color:red">ATTENTION: OpenAI Gym is depecrated and is moved to [Gymnasium](https://gymnasium.farama.org). However, due to the current changes in various versions of Gymnasium which breaks a lot of compatibility, this repository does not support Gymnasium yet. Therefore, please refer to the [OpenAI Gym](https://www.gymlibrary.dev) documentation.</span>

# Installation
The installation can be made using the Conda package manager. If you do not have Conda installed, you can download it from [here](https://docs.conda.io/en/latest/miniconda.html).
## Basic

You can install the required packages by using the conda environment file provided in the repository. To do so, run the following command in the repository directory:

``` conda env create -f environment.yml ```

This will create a conda environment named *PacmanOpenAIGym_FHDortmund* with all the required packages installed.

Make sure to activate the environment before running the game:

``` conda activate PacmanOpenAIGym_FHDortmund ```

## Additional packages
If you require additional packages, you can install them using the conda package manager by using the following command:

``` conda install <package_name> ```

If you want to install a package that is not available in the conda repository, you can use the pip package manager:

``` pip install <package_name> ```

# Usage
The main file is *PacmanAgent.py* which is located in the *gym-pacman-environment* directory. To run the game, you can use the following command within the conda environment:

``` python PacmanAgent.py ```

This will run the game with the default settings. You can change the level by setting the global variable *level* to the desired level. 

The available levels are:
* RL01 - Straight tunnel
* RL02 - Square tunnel with a hunter ghost
* RL03 - Square tunnel with a random ghost
* RL04 - Square tunnel with dead ends and a hunter ghost
* RL05 - Intersecting tunnels with a hunter and a random ghost
* RL06 - Intersecting tunnels with dead ends and a hunter and a random ghost

You find the level files in the *Pacman_Level* directory.
