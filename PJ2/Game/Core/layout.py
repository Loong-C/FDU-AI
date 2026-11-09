# layout.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: This adaptation is based on AI projects developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from Game.Core.utilities import manhattanDistance
from Game.Core.game import Grid
import random
from functools import reduce

VISIBILITY_MATRIX_CACHE = {}

class Layout:
    """
    A Layout manages the static information about the game board.
    """

    def __init__(self, layoutText):
        if not layoutText or not layoutText[0]:
            raise ValueError('Layout must contain a nonempty first row')
        self.width = len(layoutText[0])
        if any(row[self.width:].strip('% \t') for row in layoutText):
            raise ValueError('Layout rows exceed the width of the first row')
        # Some bundled maps have shorter border rows or trailing spaces.
        # Missing cells are outside the board and therefore treated as walls.
        layoutText = [row[:self.width].ljust(self.width, '%') for row in layoutText]
        self.height = len(layoutText)
        self.walls = Grid(self.width, self.height, False)
        self.coins = Grid(self.width, self.height, False)
        self.swords = []
        self.agentPositions = []
        self.numDragons = 0
        self.processLayoutText(layoutText)
        self.layoutText = layoutText
        self.totalCoins = len(self.coins.asList())
        # self.initializeVisibilityMatrix()

    def getNumDragons(self):
        return self.numDragons

    def initializeVisibilityMatrix(self):
        global VISIBILITY_MATRIX_CACHE
        from Game.Core.game import Directions
        key = tuple(self.layoutText)
        if key not in VISIBILITY_MATRIX_CACHE:
            vectors = {Directions.NORTH: (0, 0.5), Directions.SOUTH: (0, -0.5),
                       Directions.EAST: (0.5, 0), Directions.WEST: (-0.5, 0),
                       Directions.STOP: (0, 0)}
            vis = [[{direction: set() for direction in vectors}
                    for _ in range(self.height)] for _ in range(self.width)]
            for x in range(self.width):
                for y in range(self.height):
                    if self.walls[x][y]:
                        continue
                    for direction, (dx, dy) in vectors.items():
                        if direction == Directions.STOP:
                            continue
                        nx, ny = x + dx, y + dy
                        while 0 <= nx < self.width and 0 <= ny < self.height:
                            # At half-cell positions, either adjacent cell being a
                            # wall blocks this ray (also handles an open map edge).
                            from math import floor, ceil
                            if ceil(nx) >= self.width or ceil(ny) >= self.height:
                                break
                            if self.walls[floor(nx)][floor(ny)] or self.walls[ceil(nx)][ceil(ny)]:
                                break
                            vis[x][y][direction].add((nx, ny))
                            nx, ny = nx + dx, ny + dy
            VISIBILITY_MATRIX_CACHE[key] = vis
        self.visibility = VISIBILITY_MATRIX_CACHE[key]

    def isWall(self, pos):
        x, col = pos
        return self.walls[x][col]

    def getRandomLegalPosition(self):
        x = random.choice(list(range(self.width)))
        y = random.choice(list(range(self.height)))
        while self.isWall((x, y)):
            x = random.choice(list(range(self.width)))
            y = random.choice(list(range(self.height)))
        return (x, y)

    def getRandomCorner(self):
        poses = [(1, 1), (1, self.height - 2), (self.width - 2, 1),
                 (self.width - 2, self.height - 2)]
        return random.choice(poses)

    def getFurthestCorner(self, heroPos):
        poses = [(1, 1), (1, self.height - 2), (self.width - 2, 1),
                 (self.width - 2, self.height - 2)]
        dist, pos = max([(manhattanDistance(p, heroPos), p) for p in poses])
        return pos

    def isVisibleFrom(self, dragonPos, heroPos, heroDirection):
        if not hasattr(self, 'visibility'):
            self.initializeVisibilityMatrix()
        row, col = [int(x) for x in heroPos]
        return dragonPos in self.visibility[row][col][heroDirection]

    def __str__(self):
        return "\n".join(self.layoutText)

    def deepCopy(self):
        # Copy stored data rather than parsing the original text again.
        layout = object.__new__(Layout)
        layout.__dict__ = self.__dict__.copy()
        layout.walls = self.walls.copy()
        layout.coins = self.coins.copy()
        layout.swords = self.swords[:]
        layout.agentPositions = self.agentPositions[:]
        layout.layoutText = self.layoutText[:]
        if hasattr(self, 'visibility'):
            from copy import deepcopy
            layout.visibility = deepcopy(self.visibility)
        return layout

    def processLayoutText(self, layoutText):
        """
        Coordinates are flipped from the input format to the (x,y) convention here

        The shape of the maze.  Each character
        represents a different type of object.
         % - Wall
         . - Coin
         o - Sword
         G - Dragon (layout symbol retained for compatibility)
         P - Hero (layout symbol retained for compatibility)
        Other characters are ignored.
        """
        maxY = self.height - 1
        for y in range(self.height):
            for x in range(self.width):
                layoutChar = layoutText[maxY - y][x]
                self.processLayoutChar(x, y, layoutChar)
        self.agentPositions.sort()
        self.agentPositions = [(i == 0, pos) for i, pos in self.agentPositions]

    def processLayoutChar(self, x, y, layoutChar):
        if layoutChar == '%':
            self.walls[x][y] = True
        elif layoutChar == '.':
            self.coins[x][y] = True
        elif layoutChar == 'o':
            self.swords.append((x, y))
        elif layoutChar == 'P':
            self.agentPositions.append((0, (x, y)))
        elif layoutChar in ['G']:
            self.agentPositions.append((1, (x, y)))
            self.numDragons += 1
        elif layoutChar in ['1', '2', '3', '4']:
            self.agentPositions.append((int(layoutChar), (x, y)))
            self.numDragons += 1
