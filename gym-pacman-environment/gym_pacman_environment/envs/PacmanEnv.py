import os
import random
from builtins import print

import gym
import numpy as np
import pygame
from gym import spaces

import AStar
import PacmanAgent

# The number of actions after every ghost waits for 1 turn
# Default 5
GHOST_SPEED = 5

# TODO delete
random.seed(1)


class Pacman(object):
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type


class Ghost(object):
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type
        self.actions = 0
        self.dotOccupied = False


class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):  # TODO set to 4
        self.action_space = spaces.Discrete(4)  # up, left, right, down

        self.turns = 0
        self.totalDots = 0
        self.remainingDots = 0
        self.view = []
        self.entities = []

        self.agentType = None

        self.reset()

        # a 2D array of width x height of all possible tiles of the gameworld, based on PacmanAgent.tileTypes
        # The index of each tileType will represent the respective tileType, e.G. 0 = empty, 1 = wall etc.
        self.observation_space = spaces.Box(low=0, high=5, shape=(len(self.view), len(self.view[0])), dtype=np.int32)

        # Needed for rendering
        # Get current path
        self.path = os.path.dirname(os.path.abspath(__file__))

        self.img_path = self.path + "/../img/"
        self.pacman_img = pygame.image.load(self.img_path + "pacman_red_MyAgent.png")
        self.ghost_rnd_img = pygame.image.load(self.img_path + "ghost_blue_SIMPLE_RANDOM.png")
        self.ghost_hunter_img = pygame.image.load(self.img_path + "hw_ghost_hunter.png")
        self.dot_img = pygame.image.load(self.img_path + "dot.png")
        self.tile_size = 50
        self.clock = None
        self.window = None
        self.window_size = 350

    def countDots(self, view):
        dots = 0
        for x in range(0, len(view)):
            for y in range(0, len(view[x])):
                if view[x][y] == PacmanAgent.tileTypes["dot"]:
                    dots = dots + 1
        return dots

    def step(self, action):
        game_over = False
        # Move the pacman via the selected action
        for e in self.entities:
            if e.type == "pacman":
                game_over = self.updatePacman(e, action)
                break

        self.turns = self.turns + 1
        reward = 1 - (self.remainingDots / self.totalDots)
        # The game is over if all dots have been eaten
        game_over = self.remainingDots == 0 or game_over

        # Let all ghosts move and check if any ghost eats the pacman
        pacmanEaten = False
        for e in self.entities:
            if e.type == "ghost_hunter":
                pacmanEaten = self.updateGhostHunter(e) or pacmanEaten
            if e.type == "ghost_rnd":
                pacmanEaten = self.updateGhostRandom(e) or pacmanEaten
        game_over = pacmanEaten or game_over

        # Calculate the reward for the ghost
        if self.agentType == "Ghost":
            # If the game is over and not all dots have been eaten, then the ghost has won
            if game_over and self.remainingDots > 0:
                reward = 1
            else:
                reward = 1 - reward

        if not PacmanAgent.usingSimpleObservations:
            observations = self.convert_tiletypes(self.view)

        # if PacmanAgent.printState:
        #     print("Turn:", self.turns, "Reward:", reward, "Gameover:", gameover)
        #     print("Remaining Dots:", self.remainingDots, "Total Dots:", self.totalDots)
        #     # if PacmanAgent.usingPythonAgent:
        #     #     for row in self.view:
        #     #         s = ""
        #     #         for cell in row:
        #     #             s = s + cell
        #     #         print(s)
        #     # else:
        #     for y in range(len(self.view[0])):
        #         s = ""
        #         for x in range(len(self.view)):
        #             s = s + self.view[x][y]
        #         print(s)

        if PacmanAgent.usingSimpleObservations:
            return self.view, reward, game_over, {}
        else:
            return observations, reward, game_over, {}

    def convert_tiletypes(self, view=None):
        observations = np.empty((len(self.view), len(self.view[0])), dtype=np.int32)
        for y in range(0, len(view[0])):
            for x in range(len(view)):
                i = 0
                for t in PacmanAgent.tileTypes:
                    if PacmanAgent.tileTypes[t] == view[x][y]:
                        observations[x][y] = i
                        break
                    i += 1
        return observations

    def movementOffsets(self, action):
        if action == "GO_NORTH":
            xOffset = 0
            yOffset = -1
        elif action == "GO_WEST":
            xOffset = -1
            yOffset = 0
        elif action == "GO_EAST":
            xOffset = 1
            yOffset = 0
        elif action == "GO_SOUTH":
            xOffset = 0
            yOffset = 1
        elif action == "WAIT":
            xOffset = 0
            yOffset = 0
        else:
            raise Exception("Unhandled Case " + action)
        return xOffset, yOffset

    def outOfBounds(self, offsets, entity):
        if entity.x + offsets[0] < 0 or entity.x + offsets[0] > len(self.view) - 1:
            return True
        if entity.y + offsets[1] < 0 or entity.y + offsets[1] > len(self.view[0]) - 1:
            return True

    def updatePacman(self, pacman, action):
        # 0 = up, 1 = left, 2 = right, 3 = down
        offsets = self.movementOffsets(action)
        if self.outOfBounds(offsets, pacman):
            return False
        destination = self.view[pacman.x + offsets[0]][pacman.y + offsets[1]]
        if destination == PacmanAgent.tileTypes["wall"]:
            return False
        elif destination == PacmanAgent.tileTypes["dot"]:
            self.view[pacman.x][pacman.y] = PacmanAgent.tileTypes["empty"]
            pacman.x = pacman.x + offsets[0]
            pacman.y = pacman.y + offsets[1]
            self.view[pacman.x][pacman.y] = PacmanAgent.tileTypes["pacman"]
            self.remainingDots = self.remainingDots - 1
            return self.remainingDots == 0
        elif destination == PacmanAgent.tileTypes["empty"]:
            self.view[pacman.x][pacman.y] = PacmanAgent.tileTypes["empty"]
            pacman.x = pacman.x + offsets[0]
            pacman.y = pacman.y + offsets[1]
            self.view[pacman.x][pacman.y] = PacmanAgent.tileTypes["pacman"]
            return False
        elif destination == PacmanAgent.tileTypes["ghost_hunter"] or destination == PacmanAgent.tileTypes["ghost_rnd"]:
            self.view[pacman.x][pacman.y] = PacmanAgent.tileTypes["empty"]
            return True
        raise Exception("Invalid Destination:", destination)

    def updateGhostHunter(self, ghost):
        ghost.actions = ghost.actions + 1
        if ghost.actions % GHOST_SPEED == 0:
            return False

        action = self.getActionToHuntPacman(ghost)

        if action == 0:
            action = "GO_NORTH"
        elif action == 1:
            action = "GO_EAST"
        elif action == 2:
            action = "GO_SOUTH"
        elif action == 3:
            action = "GO_WEST"
        elif action == 4:
            # Do nothing
            return False

        # Execute the next action
        offsets = self.movementOffsets(action)
        destination = [ghost.x + offsets[0], ghost.y + offsets[1]]

        return self.moveGhost(ghost, destination)

    def getActionToHuntPacman(self, ghost):
        start = [ghost.x, ghost.y]
        goal = None
        for e in self.entities:
            if e.type == "pacman":
                goal = e
                break
        if goal is None:
            return 4

        passableFilter = lambda t, x, y: t != PacmanAgent.tileTypes["wall"] and t != PacmanAgent.tileTypes[
            "ghost_rnd"] and t != PacmanAgent.tileTypes["ghost_hunter"]
        pacman = None
        for e in self.entities:
            if e.type == "pacman":
                pacman = e
        path = AStar.aStar(self.view, ghost.x, ghost.y, pacman.x, pacman.y, passableFilter)

        # print("gxy:", ghost.x, "", ghost.y, "to pxy:", pacman.x, "", pacman.y)
        # print("path:", path)

        # Return the next action to get closer to the pacman or wait if no action if possible
        if path is None:
            return 4
        else:
            v = path.pop(0)
            if v[1] < start[1]:
                return 0
            elif v[0] > start[0]:
                return 1
            elif v[1] > start[1]:
                return 2
            else:
                return 3

    def updateGhostRandom(self, ghost):
        while True:
            ghost.actions = ghost.actions + 1
            if ghost.actions % GHOST_SPEED == 0:
                return False

            # Determine the next action
            x, y = ghost.x, ghost.y
            action = "WAIT"
            r = random.randrange(0, 5, 1)
            limit = 30
            while limit > 0:
                if not self.outOfBounds(offsets=self.movementOffsets("GO_NORTH"), entity=ghost) and (r == 0) and self.view[x][y - 1] != PacmanAgent.tileTypes["wall"]:
                    action = "GO_NORTH"
                    break
                if not self.outOfBounds(offsets=self.movementOffsets("GO_EAST"), entity=ghost) and r == 1 and self.view[x + 1][y] != PacmanAgent.tileTypes["wall"]:
                    action = "GO_EAST"
                    break
                if not self.outOfBounds(offsets=self.movementOffsets("GO_SOUTH"), entity=ghost) and r == 2 and self.view[x][y + 1] != PacmanAgent.tileTypes["wall"]:
                    action = "GO_SOUTH"
                    break
                if not self.outOfBounds(offsets=self.movementOffsets("GO_WEST"), entity=ghost) and r == 3 and self.view[x - 1][y] != PacmanAgent.tileTypes["wall"]:
                    action = "GO_WEST"
                    break
                if r == 4:
                    # Do nothing
                    return False
                r = random.randrange(0, 5, 1)
                limit = limit - 1
            # Execute the next action
            offsets = self.movementOffsets(action)
            if self.outOfBounds(offsets, ghost):
                continue
            destination = [ghost.x + offsets[0], ghost.y + offsets[1]]
            if self.view[destination[0]][destination[1]] == PacmanAgent.tileTypes["wall"]:
                continue
            return self.moveGhost(ghost, destination)

    def moveGhost(self, ghost, destination):
        destType = self.view[destination[0]][destination[1]]
        if destType == PacmanAgent.tileTypes["wall"]:
            return False
        elif destType == PacmanAgent.tileTypes["dot"] or destType == PacmanAgent.tileTypes["empty"] or destType == \
                PacmanAgent.tileTypes["pacman"]:
            if ghost.dotOccupied:
                self.view[ghost.x][ghost.y] = PacmanAgent.tileTypes["dot"]
            else:
                self.view[ghost.x][ghost.y] = PacmanAgent.tileTypes["empty"]
            ghost.x = destination[0]
            ghost.y = destination[1]
            ghost.dotOccupied = self.view[ghost.x][ghost.y] == PacmanAgent.tileTypes["dot"]
            self.view[ghost.x][ghost.y] = PacmanAgent.tileTypes[ghost.type]
            if destType == PacmanAgent.tileTypes["pacman"]:
                return True
            else:
                return False
        elif destType == PacmanAgent.tileTypes["ghost_hunter"] or destType == PacmanAgent.tileTypes["ghost_rnd"]:
            return False

    def reset(self):
        self.view = []
        self.entities = []

        level = PacmanAgent.level
        self.view = [[level[x][y] for y in range(len(level[0]))] for x in range(len(level))]
        for y in range(0, len(level[0])):
            for x in range(len(level)):
                if level[x][y] == PacmanAgent.tileTypes["ghost_hunter"]:
                    self.entities.append(Ghost(x, y, "ghost_hunter"))
                elif level[x][y] == PacmanAgent.tileTypes["ghost_rnd"]:
                    self.entities.append(Ghost(x, y, "ghost_rnd"))
                elif level[x][y] == PacmanAgent.tileTypes["pacman"]:
                    self.entities.append(Pacman(x, y, "pacman"))

        self.totalDots = self.countDots(self.view)
        self.remainingDots = self.totalDots
        self.turns = 0

        return np.array(self.convert_tiletypes(self.view))

    def render(self, mode='human'):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))

        # Draw the grid with the dots, the ghosts and the pacman. Use the images from the resources folder
        for y in range(0, len(self.view[0])):
            for x in range(len(self.view)):
                # Make wall black
                if self.view[y][x] == PacmanAgent.tileTypes["wall"]:
                    pygame.draw.rect(canvas, (255, 255, 255), (x * self.tile_size, y * self.tile_size, self.tile_size,
                                                               self.tile_size))
                elif self.view[y][x] == PacmanAgent.tileTypes["dot"]:
                    canvas.blit(self.dot_img, (x * self.tile_size, y * self.tile_size))
                elif self.view[y][x] == PacmanAgent.tileTypes["ghost_hunter"]:
                    canvas.blit(self.ghost_hunter_img, (x * self.tile_size, y * self.tile_size))
                elif self.view[y][x] == PacmanAgent.tileTypes["ghost_rnd"]:
                    canvas.blit(self.ghost_rnd_img, (x * self.tile_size, y * self.tile_size))
                elif self.view[y][x] == PacmanAgent.tileTypes["pacman"]:
                    canvas.blit(self.pacman_img, (x * self.tile_size, y * self.tile_size))

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        self.clock.tick(60)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None