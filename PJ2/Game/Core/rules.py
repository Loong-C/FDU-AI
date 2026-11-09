# hero.py
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


from Game.Core.game import GameStateData, Directions, Actions
from Game.Core.utilities import nearestPoint, manhattanDistance

WEAKENED_TIME = 40
COLLISION_TOLERANCE = 0.7
TIME_PENALTY = 1

class GameState:
    """
    A GameState specifies the full game state, including the coins, swords,
    agent configurations and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.

    Note that in classic Hero, Hero is always agent 0.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    # static variable keeps track of which states have had getLegalActions called
    explored = set()

    def getAndResetExplored():
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp
    getAndResetExplored = staticmethod(getAndResetExplored)

    def getLegalActions(self, agentIndex=0):
        """
        Returns the legal actions for the agent specified.
        """
#        GameState.explored.add(self)
        if self.isWin() or self.isLose():
            return []

        if agentIndex == 0:  # Hero is moving
            return HeroRules.getLegalActions(self)
        else:
            return DragonRules.getLegalActions(self, agentIndex)

    def generateSuccessor(self, agentIndex, action):
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.isWin() or self.isLose():
            raise Exception('Can\'t generate a successor of a terminal state.')

        # Copy current state
        state = GameState(self)

        # Let agent's logic deal with its action's effects on the board
        if agentIndex == 0:  # Hero is moving
            state.data._defeated = [False for i in range(state.getNumAgents())]
            HeroRules.applyAction(state, action)
        else:                # A dragon is moving
            DragonRules.applyAction(state, action, agentIndex)

        # Time passes
        if agentIndex == 0:
            state.data.scoreChange += -TIME_PENALTY  # Penalty for waiting around
        else:
            DragonRules.decrementTimer(state.data.agentStates[agentIndex])

        # Resolve multi-agent effects
        DragonRules.checkDeath(state, agentIndex)

        # Book keeping
        state.data._agentMoved = agentIndex
        state.data.score += state.data.scoreChange
        if self._trackExplored:
            GameState.explored.add(self)
            GameState.explored.add(state)
        return state

    def getLegalHeroActions(self):
        return self.getLegalActions(0)

    def generateHeroSuccessor(self, action):
        """
        Generates the successor state after the specified hero move
        """
        return self.generateSuccessor(0, action)

    def getHeroState(self):
        """
        Returns an AgentState object for hero (in Game/Core/game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        return self.data.agentStates[0].copy()

    def getHeroPosition(self):
        return self.data.agentStates[0].getPosition()

    def getDragonStates(self):
        return self.data.agentStates[1:]

    def getDragonState(self, agentIndex):
        if agentIndex <= 0 or agentIndex >= self.getNumAgents():
            raise Exception("Invalid index passed to getDragonState")
        return self.data.agentStates[agentIndex]

    def getDragonPosition(self, agentIndex):
        if agentIndex <= 0 or agentIndex >= self.getNumAgents():
            raise Exception("Invalid index passed to getDragonPosition")
        return self.data.agentStates[agentIndex].getPosition()

    def getDragonPositions(self):
        return [s.getPosition() for s in self.getDragonStates()]

    def getNumAgents(self):
        return len(self.data.agentStates)

    def getScore(self):
        return float(self.data.score)

    def getSwords(self):
        """
        Returns a list of positions (x,y) of the remaining swords.
        """
        return self.data.swords

    def getNumCoins(self):
        return self.data.coins.count()

    def getCoins(self):
        """
        Returns a Grid of boolean coins indicator variables.

        Grids can be accessed via list notation, so to check
        if there are coins at (x,y), just call

        currentCoin = state.getCoins()
        if currentCoin[x][y] == True: ...
        """
        return self.data.coins

    def getWalls(self):
        """
        Returns a Grid of boolean wall indicator variables.

        Grids can be accessed via list notation, so to check
        if there is a wall at (x,y), just call

        walls = state.getWalls()
        if walls[x][y] == True: ...
        """
        return self.data.layout.walls

    def hasCoin(self, x, y):
        return self.data.coins[x][y]

    def hasWall(self, x, y):
        return self.data.layout.walls[x][y]

    def isLose(self):
        return self.data._lose

    def isWin(self):
        return self.data._win

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prevState=None):
        """
        Generates a new state by copying information from its predecessor.
        """
        self._trackExplored = getattr(prevState, '_trackExplored', True)
        if prevState != None:  # Initial state
            self.data = GameStateData(prevState.data)
        else:
            self.data = GameStateData()

    def deepCopy(self):
        state = GameState(self)
        state.data = self.data.deepCopy()
        return state

    def __eq__(self, other):
        """
        Allows two states to be compared.
        """
        return hasattr(other, 'data') and self.data == other.data

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        """
        return hash(self.data)

    def __str__(self):

        return str(self.data)

    def initialize(self, layout, numDragonAgents=1000):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.data.initialize(layout, numDragonAgents)


class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout


    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if state.isWin():
            self.win(state, game)
        if state.isLose():
            self.lose(state, game)

    def win(self, state, game):
        if not self.quiet:
            game.report("Hero emerges victorious! Score: %d" % state.data.score)
        game.gameOver = True

    def lose(self, state, game):
        if not self.quiet:
            game.report("Hero was defeated! Score: %d" % state.data.score)
        game.gameOver = True

    def getProgress(self, game):
        initial = self.initialState.getNumCoins()
        return float(game.state.getNumCoins()) / initial if initial else 0.0

    def agentCrash(self, game, agentIndex):
        if agentIndex == 0:
            game.report("Hero crashed")
        else:
            game.report("A dragon crashed")

    def getMaxTotalTime(self, agentIndex):
        return self.timeout

    def getMaxStartupTime(self, agentIndex):
        return self.timeout

    def getMoveWarningTime(self, agentIndex):
        return self.timeout

    def getMoveTimeout(self, agentIndex):
        return self.timeout

    def getMaxTimeWarnings(self, agentIndex):
        return 0


class HeroRules:
    """
    These functions govern how hero interacts with his environment under
    the classic game rules.
    """
    HERO_SPEED = 1

    def getLegalActions(state):
        """
        Returns a list of possible actions.
        """
        return Actions.getPossibleActions(state.getHeroState().configuration, state.data.layout.walls)
    getLegalActions = staticmethod(getLegalActions)

    def applyAction(state, action):
        """
        Edits the state to reflect the results of the action.
        """
        legal = HeroRules.getLegalActions(state)
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        heroState = state.data.agentStates[0]

        # Update Configuration
        vector = Actions.directionToVector(action, HeroRules.HERO_SPEED)
        heroState.configuration = heroState.configuration.generateSuccessor(
            vector)

        # Collect any item at the Hero's position
        next = heroState.configuration.getPosition()
        nearest = nearestPoint(next)
        if manhattanDistance(nearest, next) <= 0.5:
            # Remove the collected item
            HeroRules.collectItems(nearest, state)
    applyAction = staticmethod(applyAction)

    def collectItems(position, state):
        x, y = position
        # Collect a coin
        if state.data.coins[x][y]:
            state.data.scoreChange += 10
            state.data.coins = state.data.coins.copy()
            state.data.coins[x][y] = False
            state.data._coinCollected = position
            # TODO: cache numCoins?
            numCoins = state.getNumCoins()
            if numCoins == 0 and not state.data._lose:
                state.data.scoreChange += 500
                state.data._win = True
        # Collect a sword
        if(position in state.getSwords()):
            state.data.swords.remove(position)
            state.data._swordCollected = position
            # Reset all dragons' weakened timers
            for index in range(1, len(state.data.agentStates)):
                state.data.agentStates[index].weakenedTimer = WEAKENED_TIME
    collectItems = staticmethod(collectItems)


class DragonRules:
    """
    These functions dictate how dragons interact with their environment.
    """
    DRAGON_SPEED = 1.0

    def getLegalActions(state, dragonIndex):
        """
        Dragons cannot stop, and cannot turn around unless they
        reach a dead end, but can turn 90 degrees at intersections.
        """
        conf = state.getDragonState(dragonIndex).configuration
        possibleActions = Actions.getPossibleActions(
            conf, state.data.layout.walls)
        reverse = Actions.reverseDirection(conf.direction)
        if Directions.STOP in possibleActions:
            possibleActions.remove(Directions.STOP)
        if reverse in possibleActions and len(possibleActions) > 1:
            possibleActions.remove(reverse)
        return possibleActions
    getLegalActions = staticmethod(getLegalActions)

    def applyAction(state, action, dragonIndex):

        legal = DragonRules.getLegalActions(state, dragonIndex)
        if action not in legal:
            raise Exception("Illegal dragon action " + str(action))

        dragonState = state.data.agentStates[dragonIndex]
        speed = DragonRules.DRAGON_SPEED
        if dragonState.weakenedTimer > 0:
            speed /= 2.0
        vector = Actions.directionToVector(action, speed)
        dragonState.configuration = dragonState.configuration.generateSuccessor(
            vector)
    applyAction = staticmethod(applyAction)

    def decrementTimer(dragonState):
        timer = dragonState.weakenedTimer
        if timer == 1:
            dragonState.configuration.pos = nearestPoint(
                dragonState.configuration.pos)
        dragonState.weakenedTimer = max(0, timer - 1)
    decrementTimer = staticmethod(decrementTimer)

    def checkDeath(state, agentIndex):
        heroPosition = state.getHeroPosition()
        if agentIndex == 0:  # Hero just moved; Anyone can kill him
            for index in range(1, len(state.data.agentStates)):
                dragonState = state.data.agentStates[index]
                dragonPosition = dragonState.configuration.getPosition()
                if DragonRules.canKill(heroPosition, dragonPosition):
                    DragonRules.collide(state, dragonState, index)
        else:
            dragonState = state.data.agentStates[agentIndex]
            dragonPosition = dragonState.configuration.getPosition()
            if DragonRules.canKill(heroPosition, dragonPosition):
                DragonRules.collide(state, dragonState, agentIndex)
    checkDeath = staticmethod(checkDeath)

    def collide(state, dragonState, agentIndex):
        if dragonState.weakenedTimer > 0:
            state.data.scoreChange += 200
            DragonRules.placeDragon(state, dragonState)
            dragonState.weakenedTimer = 0
            # Added for first-person
            state.data._defeated[agentIndex] = True
        else:
            if not state.data._win and not state.data._lose:
                state.data.scoreChange -= 500
                state.data._lose = True
    collide = staticmethod(collide)

    def canKill(heroPosition, dragonPosition):
        return manhattanDistance(dragonPosition, heroPosition) <= COLLISION_TOLERANCE
    canKill = staticmethod(canKill)

    def placeDragon(state, dragonState):
        dragonState.configuration = dragonState.start
    placeDragon = staticmethod(placeDragon)
