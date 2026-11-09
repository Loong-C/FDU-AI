# graphicsDisplay.py
# ------------------
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


from graphicsUtils import *
import math
import time
import os
from game import Directions

# Path to the Hero walk animation GIFs
HERO_GIF_PATH = os.path.join(os.path.dirname(__file__), "images", "walk.gif")
HERO_GIF_BACK_PATH = os.path.join(os.path.dirname(__file__), "images", "walk_back.gif")

# Path to wall texture image
WALL_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images", "wall.png")

# Path to the coin animation GIF
COIN_GIF_PATH = os.path.join(os.path.dirname(__file__), "images", "coin.gif")

# Path to sword (power-up) image
SWORD_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images", "sword.png")

# Paths to the first dragon's animation GIFs (red dragon)
DRAGON_GIF_BACK = os.path.join(os.path.dirname(__file__), "images", "dragon_back.gif")
DRAGON_GIF_HEAD = os.path.join(os.path.dirname(__file__), "images", "dragon_head.gif")
DRAGON_GIF_LEFT = os.path.join(os.path.dirname(__file__), "images", "dragon_left.gif")
DRAGON_GIF_RIGHT = os.path.join(os.path.dirname(__file__), "images", "dragon_right.gif")

# Paths to the second dragon's animation GIFs (green dragon)
DRAGON_GIF_BACK_2 = os.path.join(os.path.dirname(__file__), "images", "dragon_back_2.gif")
DRAGON_GIF_HEAD_2 = os.path.join(os.path.dirname(__file__), "images", "dragon_head_2.gif")
DRAGON_GIF_LEFT_2 = os.path.join(os.path.dirname(__file__), "images", "dragon_left_2.gif")
DRAGON_GIF_RIGHT_2 = os.path.join(os.path.dirname(__file__), "images", "dragon_right_2.gif")

###########################
#  GRAPHICS DISPLAY CODE  #
###########################

# Most code by Dan Klein and John Denero written or rewritten for cs188, UC Berkeley.
# Some display code from the original implementation is used / modified with permission.

DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 35
BACKGROUND_COLOR = formatColor(0, 0, 0)
WALL_COLOR = formatColor(0.0/255.0, 51.0/255.0, 255.0/255.0)
INFO_PANE_COLOR = formatColor(.4, .4, 0)
SCORE_COLOR = formatColor(.9, .9, .9)
HERO_OUTLINE_WIDTH = 2
HERO_CAPTURE_OUTLINE_WIDTH = 4

DRAGON_COLORS = []
DRAGON_COLORS.append(formatColor(.9, 0, 0))  # Red
DRAGON_COLORS.append(formatColor(0, .3, .9))  # Blue
DRAGON_COLORS.append(formatColor(.98, .41, .07))  # Orange
DRAGON_COLORS.append(formatColor(.1, .75, .7))  # Green
DRAGON_COLORS.append(formatColor(1.0, 0.6, 0.0))  # Yellow
DRAGON_COLORS.append(formatColor(.4, 0.13, 0.91))  # Purple

TEAM_COLORS = DRAGON_COLORS[:2]

DRAGON_SHAPE = [
    (0,     0.3),
    (0.25,  0.75),
    (0.5,   0.3),
    (0.75,  0.75),
    (0.75,  -0.5),
    (0.5,   -0.75),
    (-0.5,  -0.75),
    (-0.75, -0.5),
    (-0.75, 0.75),
    (-0.5,  0.3),
    (-0.25, 0.75)
]
DRAGON_SIZE = 0.65
WEAKENED_COLOR = formatColor(1, 1, 1)

DRAGON_VEC_COLORS = list(map(colorToVector, DRAGON_COLORS))

HERO_COLOR = formatColor(255.0/255.0, 255.0/255.0, 61.0/255)
HERO_SCALE = 0.5
# hero_speed = 0.25

# Coin
COIN_COLOR = formatColor(1, 1, 1)
COIN_SIZE = 0.1

# Laser
LASER_COLOR = formatColor(1, 0, 0)
LASER_SIZE = 0.02

# Sword graphics
SWORD_COLOR = formatColor(1, 1, 1)
SWORD_SIZE = 0.25

# Drawing walls
WALL_RADIUS = 0.15


class InfoPane:
    def __init__(self, layout, gridSize):
        self.gridSize = gridSize
        self.width = (layout.width) * gridSize
        self.base = (layout.height + 1) * gridSize
        self.height = INFO_PANE_HEIGHT
        self.fontSize = 24
        self.textColor = HERO_COLOR
        self.drawPane()

    def toScreen(self, pos, y=None):
        """
          Translates a point relative from the bottom left of the info pane.
        """
        if y == None:
            x, y = pos
        else:
            x = pos

        x = self.gridSize + x  # Margin
        y = self.base + y
        return x, y

    def drawPane(self):
        self.scoreText = text(self.toScreen(
            0, 0), self.textColor, "SCORE:    0", "Times", self.fontSize, "bold")

    def initializeDragonDistances(self, distances):
        self.dragonDistanceText = []

        size = 20
        if self.width < 240:
            size = 12
        if self.width < 160:
            size = 10

        for i, d in enumerate(distances):
            t = text(self.toScreen(self.width/2 + self.width/8 * i, 0),
                     DRAGON_COLORS[i+1], d, "Times", size, "bold")
            self.dragonDistanceText.append(t)

    def updateScore(self, score):
        changeText(self.scoreText, "SCORE: % 4d" % score)

    def setTeam(self, isBlue):
        text = "RED TEAM"
        if isBlue:
            text = "BLUE TEAM"
        self.teamText = text(self.toScreen(
            300, 0), self.textColor, text, "Times", self.fontSize, "bold")

    def updateDragonDistances(self, distances):
        if len(distances) == 0:
            return
        if 'dragonDistanceText' not in dir(self):
            self.initializeDragonDistances(distances)
        else:
            for i, d in enumerate(distances):
                changeText(self.dragonDistanceText[i], d)

    def drawDragon(self):
        pass

    def drawHero(self):
        pass

    def drawWarning(self):
        pass

    def clearIcon(self):
        pass

    def updateMessage(self, message):
        pass

    def clearMessage(self):
        pass


class HeroGraphics:
    def __init__(self, zoom=1.0, frameTime=0.0, capture=False):
        self.have_window = 0
        self.currentDragonImages = {}
        self.heroImage = None
        self.zoom = zoom
        self.gridSize = DEFAULT_GRID_SIZE * zoom
        self.capture = capture
        self.frameTime = frameTime

    def checkNullDisplay(self):
        return False

    def initialize(self, state, isBlue=False):
        self.isBlue = isBlue
        self.startGraphics(state)

        # self.drawDistributions(state)
        self.distributionImages = None  # Initialized lazily
        self.drawStaticObjects(state)
        self.drawAgentObjects(state)

        # Information
        self.previousState = state

    def startGraphics(self, state):
        self.layout = state.layout
        layout = self.layout
        self.width = layout.width
        self.height = layout.height
        self.make_window(self.width, self.height)
        self.infoPane = InfoPane(layout, self.gridSize)
        self.currentState = layout

    def drawDistributions(self, state):
        walls = state.layout.walls
        dist = []
        for x in range(walls.width):
            distx = []
            dist.append(distx)
            for y in range(walls.height):
                (screen_x, screen_y) = self.to_screen((x, y))
                block = square((screen_x, screen_y),
                               0.5 * self.gridSize,
                               color=BACKGROUND_COLOR,
                               filled=1, behind=2)
                distx.append(block)
        self.distributionImages = dist

    def drawStaticObjects(self, state):
        layout = self.layout
        self.drawWalls(layout.walls)
        self.coins = self.drawCoins(layout.coins)
        self.swords = self.drawSwords(layout.swords)
        refresh()

    def drawAgentObjects(self, state):
        self.agentImages = []  # (agentState, image)
        for index, agent in enumerate(state.agentStates):
            if agent.isHero:
                image = self.drawHero(agent, index)
                self.agentImages.append((agent, image))
            else:
                image = self.drawDragon(agent, index)
                self.agentImages.append((agent, image))
        refresh()

    def swapImages(self, agentIndex, newState):
        """
          Changes an image from a dragon to a hero or vis versa (for capture)
        """
        prevState, prevImage = self.agentImages[agentIndex]
        for item in prevImage:
            remove_from_screen(item)
        if newState.isHero:
            image = self.drawHero(newState, agentIndex)
            self.agentImages[agentIndex] = (newState, image)
        else:
            image = self.drawDragon(newState, agentIndex)
            self.agentImages[agentIndex] = (newState, image)
        refresh()

    def update(self, newState):
        agentIndex = newState._agentMoved
        agentState = newState.agentStates[agentIndex]

        if self.agentImages[agentIndex][0].isHero != agentState.isHero:
            self.swapImages(agentIndex, agentState)
        prevState, prevImage = self.agentImages[agentIndex]
        if agentState.isHero:
            self.animateHero(agentState, prevState, prevImage)
        else:
            self.moveDragon(agentState, agentIndex, prevState, prevImage)
        self.agentImages[agentIndex] = (agentState, prevImage)

        for dragonIndex, dragonState in enumerate(newState.agentStates[1:], start=1):
            previousDragon, imageParts = self.agentImages[dragonIndex]
            if self.dragonWeakened.get(dragonIndex) != (dragonState.weakenedTimer > 0):
                self.moveDragon(dragonState, dragonIndex, previousDragon, imageParts)
                self.agentImages[dragonIndex] = (dragonState, imageParts)

        if newState._coinCollected != None:
            self.removeCoin(newState._coinCollected, self.coins)
        if newState._swordCollected != None:
            self.removeSword(newState._swordCollected, self.swords)
        self.infoPane.updateScore(newState.score)
        if 'dragonDistances' in dir(newState):
            self.infoPane.updateDragonDistances(newState.dragonDistances)

    def make_window(self, width, height):
        grid_width = (width-1) * self.gridSize
        grid_height = (height-1) * self.gridSize
        screen_width = 2*self.gridSize + grid_width
        screen_height = 2*self.gridSize + grid_height + INFO_PANE_HEIGHT

        begin_graphics(screen_width,
                       screen_height,
                       BACKGROUND_COLOR,
                       "Hero vs Dragons")

    def drawHero(self, hero, index):
        position = self.getPosition(hero)
        screen_point = self.to_screen(position)
        direction = self.getDirection(hero)
        
        # Use GIF animation for Hero instead of circle
        # GIF is 6x11 pixels, scale it to roughly 80% of grid size
        scale = (self.gridSize * 0.8) / 11.0  # Scale based on GIF height (11 pixels)
        
        # Choose GIF based on direction - use back animation when facing North (up/w)
        gif_path = HERO_GIF_BACK_PATH if direction == 'North' else HERO_GIF_PATH
        
        img_id = create_gif_image(screen_point, gif_path, scale)
        # Start the animation
        animate_gif(img_id, delay=150)
        
        # Store current direction for later comparison
        self.heroDirection = direction
        
        return [img_id]

    def getEndpoints(self, direction, position=(0, 0)):
        x, y = position
        pos = x - int(x) + y - int(y)
        width = 30 + 80 * math.sin(math.pi * pos)

        delta = width / 2
        if (direction == 'West'):
            endpoints = (180+delta, 180-delta)
        elif (direction == 'North'):
            endpoints = (90+delta, 90-delta)
        elif (direction == 'South'):
            endpoints = (270+delta, 270-delta)
        else:
            endpoints = (0+delta, 0-delta)
        return endpoints

    def moveHero(self, position, direction, image):
        screenPosition = self.to_screen(position)
        
        # Check if direction changed and need to switch animation
        prev_direction = getattr(self, 'heroDirection', None)
        need_switch = (direction == 'North') != (prev_direction == 'North')
        
        if need_switch:
            # Remove old image and create new one with correct animation
            remove_from_screen(image[0])
            scale = (self.gridSize * 0.8) / 11.0
            gif_path = HERO_GIF_BACK_PATH if direction == 'North' else HERO_GIF_PATH
            new_img_id = create_gif_image(screenPosition, gif_path, scale)
            animate_gif(new_img_id, delay=150)
            image[0] = new_img_id
            self.heroDirection = direction
        else:
            # Just move the GIF image
            move_gif_image(image[0], screenPosition)
        
        refresh()

    def animateHero(self, hero, prevHero, image):
        if self.frameTime < 0:
            print('Press any key to step forward, "q" to play')
            keys = wait_for_keys()
            if 'q' in keys:
                self.frameTime = 0.1
        if self.frameTime > 0.01 or self.frameTime < 0:
            start = time.time()
            fx, fy = self.getPosition(prevHero)
            px, py = self.getPosition(hero)
            frames = 4.0
            for i in range(1, int(frames) + 1):
                pos = px*i/frames + fx * \
                    (frames-i)/frames, py*i/frames + fy*(frames-i)/frames
                self.moveHero(pos, self.getDirection(hero), image)
                refresh()
                sleep(abs(self.frameTime) / frames)
        else:
            self.moveHero(self.getPosition(hero),
                            self.getDirection(hero), image)
        refresh()

    def getDragonColor(self, dragon, dragonIndex):
        if dragon.weakenedTimer > 0:
            return WEAKENED_COLOR
        else:
            return DRAGON_COLORS[dragonIndex]

    def getDragonGifPath(self, direction, dragonIndex=1):
        """Get the appropriate dragon GIF based on direction and dragon index"""
        # dragonIndex 1 = first dragon (red), dragonIndex 2 = second dragon (green)
        if dragonIndex == 2:
            # Second dragon - green dragon
            if direction == 'North':
                return DRAGON_GIF_BACK_2
            elif direction == 'South':
                return DRAGON_GIF_HEAD_2
            elif direction == 'West':
                return DRAGON_GIF_LEFT_2
            else:  # East or Stop
                return DRAGON_GIF_RIGHT_2
        else:
            # First dragon (and others) - red dragon
            if direction == 'North':
                return DRAGON_GIF_BACK
            elif direction == 'South':
                return DRAGON_GIF_HEAD
            elif direction == 'West':
                return DRAGON_GIF_LEFT
            else:  # East or Stop
                return DRAGON_GIF_RIGHT

    def drawDragon(self, dragon, agentIndex):
        pos = self.getPosition(dragon)
        direction = self.getDirection(dragon)
        screen_point = self.to_screen(pos)
        
        # Use dragon GIF animation instead of polygon
        # Dragon GIF is 16x16 pixels, scale to about 80% of grid size
        scale = (self.gridSize * 0.8) / 16.0
        
        gif_path = self.getDragonGifPath(direction, agentIndex)
        img_id = create_gif_image(screen_point, gif_path, scale,
                                  opacity=0.5 if dragon.weakenedTimer > 0 else 1.0)
        animate_gif(img_id, delay=150)
        
        # Store direction for this dragon
        if not hasattr(self, 'dragonDirections'):
            self.dragonDirections = {}
        self.dragonDirections[agentIndex] = direction
        if not hasattr(self, 'dragonWeakened'):
            self.dragonWeakened = {}
        self.dragonWeakened[agentIndex] = dragon.weakenedTimer > 0
        
        # Return as list for compatibility (single element now)
        return [img_id]

    def moveEyes(self, pos, dir, eyes):
        (screen_x, screen_y) = (self.to_screen(pos))
        dx = 0
        dy = 0
        if dir == 'North':
            dy = -0.2
        if dir == 'South':
            dy = 0.2
        if dir == 'East':
            dx = 0.2
        if dir == 'West':
            dx = -0.2
        moveCircle(eyes[0], (screen_x+self.gridSize*DRAGON_SIZE*(-0.3+dx/1.5), screen_y -
                             self.gridSize*DRAGON_SIZE*(0.3-dy/1.5)), self.gridSize*DRAGON_SIZE*0.2)
        moveCircle(eyes[1], (screen_x+self.gridSize*DRAGON_SIZE*(0.3+dx/1.5), screen_y -
                             self.gridSize*DRAGON_SIZE*(0.3-dy/1.5)), self.gridSize*DRAGON_SIZE*0.2)
        moveCircle(eyes[2], (screen_x+self.gridSize*DRAGON_SIZE*(-0.3+dx), screen_y -
                             self.gridSize*DRAGON_SIZE*(0.3-dy)), self.gridSize*DRAGON_SIZE*0.08)
        moveCircle(eyes[3], (screen_x+self.gridSize*DRAGON_SIZE*(0.3+dx), screen_y -
                             self.gridSize*DRAGON_SIZE*(0.3-dy)), self.gridSize*DRAGON_SIZE*0.08)

    def moveDragon(self, dragon, dragonIndex, prevDragon, dragonImageParts):
        new_pos = self.getPosition(dragon)
        new_screen = self.to_screen(new_pos)
        new_direction = self.getDirection(dragon)
        
        # Check if direction changed and need to switch animation
        prev_direction = self.dragonDirections.get(dragonIndex, None)
        weakened = dragon.weakenedTimer > 0
        
        if prev_direction != new_direction or self.dragonWeakened.get(dragonIndex) != weakened:
            # Remove old image and create new one with correct animation
            remove_from_screen(dragonImageParts[0])
            scale = (self.gridSize * 0.8) / 16.0
            gif_path = self.getDragonGifPath(new_direction, dragonIndex)
            new_img_id = create_gif_image(new_screen, gif_path, scale,
                                          opacity=0.5 if weakened else 1.0)
            animate_gif(new_img_id, delay=150)
            dragonImageParts[0] = new_img_id
            self.dragonDirections[dragonIndex] = new_direction
            self.dragonWeakened[dragonIndex] = weakened
        else:
            # Just move the GIF image
            move_gif_image(dragonImageParts[0], new_screen)
        
        refresh()

    def getPosition(self, agentState):
        if agentState.configuration == None:
            return (-1000, -1000)
        return agentState.getPosition()

    def getDirection(self, agentState):
        if agentState.configuration == None:
            return Directions.STOP
        return agentState.configuration.getDirection()

    def finish(self):
        end_graphics()

    def to_screen(self, point):
        (x, y) = point
        #y = self.height - y
        x = (x + 1)*self.gridSize
        y = (self.height - y)*self.gridSize
        return (x, y)

    # Fixes some TK issue with off-center circles
    def to_screen2(self, point):
        (x, y) = point
        #y = self.height - y
        x = (x + 1)*self.gridSize
        y = (self.height - y)*self.gridSize
        return (x, y)

    def drawWalls(self, wallMatrix):
        # Use wall image instead of drawing lines and arcs
        for xNum, x in enumerate(wallMatrix):
            for yNum, cell in enumerate(x):
                if cell:  # There's a wall here
                    pos = (xNum, yNum)
                    screen = self.to_screen(pos)
                    # Draw wall image at this position
                    create_static_image(screen, WALL_IMAGE_PATH, 
                                       self.gridSize, self.gridSize)

    def isWall(self, x, y, walls):
        if x < 0 or y < 0:
            return False
        if x >= walls.width or y >= walls.height:
            return False
        return walls[x][y]

    def drawCoins(self, coinMatrix):
        coinImages = []
        # Scale coin GIF to about 50% of grid size for coins
        scale = (self.gridSize * 0.5) / 48.0  # coin.gif is 45x48 pixels
        
        for xNum, x in enumerate(coinMatrix):
            imageRow = []
            coinImages.append(imageRow)
            for yNum, cell in enumerate(x):
                if cell:  # There is a coin here
                    screen = self.to_screen((xNum, yNum))
                    # Use animated coin GIF instead of circle
                    img_id = create_gif_image(screen, COIN_GIF_PATH, scale)
                    animate_gif(img_id, delay=100)  # Animate the coin
                    imageRow.append(img_id)
                else:
                    imageRow.append(None)
        return coinImages

    def drawSwords(self, swords):
        swordImages = {}
        # Scale sword image to about 70% of grid size
        size = self.gridSize * 0.7
        for sword in swords:
            (screen_x, screen_y) = self.to_screen(sword)
            # Use sword image instead of circle
            img_id = create_static_image((screen_x, screen_y), SWORD_IMAGE_PATH, size, size)
            swordImages[sword] = img_id
        return swordImages

    def removeCoin(self, cell, coinImages):
        x, y = cell
        remove_from_screen(coinImages[x][y])

    def removeSword(self, cell, swordImages):
        x, y = cell
        remove_from_screen(swordImages[(x, y)])

    def drawExpandedCells(self, cells):
        """
        Draws an overlay of expanded grid positions for search agents
        """
        n = float(len(cells))
        baseColor = [1.0, 0.0, 0.0]
        self.clearExpandedCells()
        self.expandedCells = []
        for k, cell in enumerate(cells):
            screenPos = self.to_screen(cell)
            cellColor = formatColor(
                *[(n-k) * c * .5 / n + .25 for c in baseColor])
            block = square(screenPos,
                           0.5 * self.gridSize,
                           color=cellColor,
                           filled=1, behind=2)
            self.expandedCells.append(block)
            if self.frameTime < 0:
                refresh()

    def clearExpandedCells(self):
        if 'expandedCells' in dir(self) and len(self.expandedCells) > 0:
            for cell in self.expandedCells:
                remove_from_screen(cell)

    def updateDistributions(self, distributions):
        "Draws an agent's belief distributions"
        # copy all distributions so we don't change their state
        distributions = [x.copy() for x in distributions]
        if self.distributionImages == None:
            self.drawDistributions(self.previousState)
        for x in range(len(self.distributionImages)):
            for y in range(len(self.distributionImages[0])):
                image = self.distributionImages[x][y]
                weights = [dist[(x, y)] for dist in distributions]

                if sum(weights) != 0:
                    pass
                # Fog of war
                color = [0.0, 0.0, 0.0]
                colors = DRAGON_VEC_COLORS[1:]  # With Hero
                if self.capture:
                    colors = DRAGON_VEC_COLORS
                for weight, gcolor in zip(weights, colors):
                    color = [min(1.0, c + 0.95 * g * weight ** .3)
                             for c, g in zip(color, gcolor)]
                changeColor(image, formatColor(*color))
        refresh()


class FirstPersonHeroGraphics(HeroGraphics):
    def __init__(self, zoom=1.0, showDragons=True, capture=False, frameTime=0):
        HeroGraphics.__init__(self, zoom, frameTime=frameTime)
        self.showDragons = showDragons
        self.capture = capture

    def initialize(self, state, isBlue=False):

        self.isBlue = isBlue
        HeroGraphics.startGraphics(self, state)
        # Initialize distribution images
        walls = state.layout.walls
        dist = []
        self.layout = state.layout

        # Draw the rest
        self.distributionImages = None  # initialize lazily
        self.drawStaticObjects(state)
        self.drawAgentObjects(state)

        # Information
        self.previousState = state

    def lookAhead(self, config, state):
        if config.getDirection() == 'Stop':
            return
        else:
            pass
            # Draw relevant dragons
            allDragons = state.getDragonStates()
            visibleDragons = state.getVisibleDragons()
            for i, dragon in enumerate(allDragons):
                if dragon in visibleDragons:
                    self.drawDragon(dragon, i)
                else:
                    self.currentDragonImages[i] = None

    def getDragonColor(self, dragon, dragonIndex):
        return DRAGON_COLORS[dragonIndex]

    def getPosition(self, dragonState):
        if not self.showDragons and not dragonState.isHero and dragonState.getPosition()[1] > 1:
            return (-1000, -1000)
        else:
            return HeroGraphics.getPosition(self, dragonState)


def add(x, y):
    return (x[0] + y[0], x[1] + y[1])


# Saving graphical output
# -----------------------
# Note: to make an animated gif from this postscript output, try the command:
# convert -delay 7 -loop 1 -compress lzw -layers optimize frame* out.gif
# convert is part of imagemagick (freeware)

SAVE_POSTSCRIPT = False
POSTSCRIPT_OUTPUT_DIR = 'frames'
FRAME_NUMBER = 0
import os


def saveFrame():
    "Saves the current graphical output as a postscript file"
    global SAVE_POSTSCRIPT, FRAME_NUMBER, POSTSCRIPT_OUTPUT_DIR
    if not SAVE_POSTSCRIPT:
        return
    if not os.path.exists(POSTSCRIPT_OUTPUT_DIR):
        os.mkdir(POSTSCRIPT_OUTPUT_DIR)
    name = os.path.join(POSTSCRIPT_OUTPUT_DIR, 'frame_%08d.ps' % FRAME_NUMBER)
    FRAME_NUMBER += 1
    writePostscript(name)  # writes the current canvas
