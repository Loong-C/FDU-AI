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


"""
hero.py holds the logic for the Hero-versus-dragons game along with the main
code to run a game.  This file is divided into three sections:

  (i)  Your interface to the quest world:
          Hero is a complex environment.  You probably don't want to
          read through all of the code we wrote to make the game runs
          correctly.  This section contains the parts of the code
          that you will need to understand in order to complete the
          project.  There is also some code in game.py that you should
          understand.

  (ii)  The hidden secrets of hero:
          This section contains all of the logic code that the hero
          environment uses to decide who can move where, who dies when
          things collide, etc.  You shouldn't need to read this section
          of code, but you can if you want.

  (iii) Framework to start a game:
          The final section contains the code for reading the command
          you use to set up the game, then starting up a new game, along with
          linking in all the external parts (agent functions, graphics).
          Check this section out to see all the options available to you.

To play your first game, type 'python hero.py' from the command line.
The keys are 'a', 's', 'd', and 'w' to move (or arrow keys).  Have fun!
"""
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
import util
import layout
import sys
import types
import time
import random
import os

###################################################
# YOUR INTERFACE TO THE HERO WORLD: A GameState #
###################################################


class GameState:
    """
    A GameState specifies the full game state, including the coins, swords,
    agent configurations and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.

    The Hero is always agent 0.
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
        Returns an AgentState object for hero (in game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        return self.data.agentStates[0].copy()

    def getHeroPosition(self):
        return self.data.agentStates[0].getPosition()

    def getDragonStates(self):
        return self.data.agentStates[1:]

    def getDragonState(self, agentIndex):
        if agentIndex == 0 or agentIndex >= self.getNumAgents():
            raise Exception("Invalid index passed to getDragonState")
        return self.data.agentStates[agentIndex]

    def getDragonPosition(self, agentIndex):
        if agentIndex == 0:
            raise Exception("Hero's index passed to getDragonPosition")
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
        if there is coins at (x,y), just call

        currentCoins = state.getCoins()
        if currentCoins[x][y] == True: ...
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

############################################################################
#                     THE HIDDEN SECRETS OF HERO                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################


WEAKENED_TIME = 40    # Moves dragons are weakened
COLLISION_TOLERANCE = 0.7  # How close dragons must be to Hero to kill
TIME_PENALTY = 1  # Number of points lost each round


class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout

    def newGame(self, layout, heroAgent, dragonAgents, display, quiet=False, catchExceptions=False):
        agents = [heroAgent] + dragonAgents[:layout.getNumDragons()]
        initState = GameState()
        initState.initialize(layout, len(dragonAgents))
        game = Game(agents, display, self, catchExceptions=catchExceptions)
        game.state = initState
        self.initialState = initState.deepCopy()
        self.quiet = quiet
        return game

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
            print("Hero is victorious! Score: %d" % state.data.score)
        game.gameOver = True

    def lose(self, state, game):
        if not self.quiet:
            print("Hero was defeated! Score: %d" % state.data.score)
        game.gameOver = True

    def getProgress(self, game):
        return float(game.state.getNumCoins()) / self.initialState.getNumCoins()

    def agentCrash(self, game, agentIndex):
        if agentIndex == 0:
            print("Hero crashed")
        else:
            print("A dragon crashed")

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
            # Remove the collected item from the board
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
            if not state.data._win:
                state.data.scoreChange -= 500
                state.data._lose = True
    collide = staticmethod(collide)

    def canKill(heroPosition, dragonPosition):
        return manhattanDistance(dragonPosition, heroPosition) <= COLLISION_TOLERANCE
    canKill = staticmethod(canKill)

    def placeDragon(state, dragonState):
        dragonState.configuration = dragonState.start
    placeDragon = staticmethod(placeDragon)

#############################
# FRAMEWORK TO START A GAME #
#############################


def default(str):
    return str + ' [Default: %default]'


def parseAgentArgs(str):
    if str == None:
        return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key, val = p, 1
        opts[key] = val
    return opts


def readCommand(argv):
    """
    Processes the command used to run hero from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python hero.py <options>
    EXAMPLES:   (1) python hero.py
                    - starts an interactive game
                (2) python hero.py --layout smallClassic --zoom 2
                OR  python hero.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default(
                          'the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='mediumClassic')
    parser.add_option('-p', '--hero', dest='hero',
                      help=default(
                          'the agent TYPE in the heroAgents module to use'),
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_option('-t', '--textGraphics', action='store_true', dest='textGraphics',
                      help='Display output as text only', default=False)
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('--animate', action='store_true', dest='animate',
                      help='Show the animated graphics window during Jev search', default=False)
    parser.add_option('-g', '--dragons', dest='dragon',
                      help=default(
                          'the dragon agent TYPE in the dragonAgents module to use'),
                      metavar='TYPE', default='RandomDragon')
    parser.add_option('-k', '--numdragons', type='int', dest='numDragons',
                      help=default('The maximum number of dragons to use'), default=4)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_option('-f', '--fixRandomSeed', action='store_true', dest='fixRandomSeed',
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('-r', '--recordActions', action='store_true', dest='record',
                      help='Writes game histories to a file (named by the time they were played)', default=False)
    parser.add_option('--replay', dest='gameToReplay',
                      help='A recorded game file (pickle) to replay', default=None)
    parser.add_option('-a', '--agentArgs', dest='agentArgs',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
                      help=default('How many episodes are training (suppresses output)'), default=0)
    parser.add_option('--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)
    parser.add_option('-c', '--catchExceptions', action='store_true', dest='catchExceptions',
                      help='Turns on exception handling and timeouts during games', default=False)
    parser.add_option('--timeout', dest='timeout', type='int',
                      help=default('Maximum length of time an agent can spend computing in a single game'), default=30)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = dict()

    # Fix the random seed
    if options.fixRandomSeed:
        random.seed('cs188')

    # Choose a layout
    args['layout'] = layout.getLayout(options.layout)
    if args['layout'] == None:
        raise Exception("The layout " + options.layout + " cannot be found")

    # Choose a Hero agent
    agentOpts = parseAgentArgs(options.agentArgs)
    heuristicName = str(agentOpts.get('heuristic', '')).lower()
    jevSearch = options.hero == 'SearchAgent' and heuristicName.startswith('jev')
    os.environ['JEV_STATUS_WINDOW'] = (
        '1' if jevSearch and not options.quietGraphics and not options.textGraphics
        else '0'
    )
    # Keep the graphics window responsive for other agents. Jev-backed search
    # is synchronous, so it uses a request-status window while planning.
    animate = options.animate or not jevSearch
    noKeyboard = options.gameToReplay == None and (
        options.textGraphics or options.quietGraphics or not animate)
    heroType = loadAgent(options.hero, noKeyboard)
    if options.numTraining > 0:
        args['numTraining'] = options.numTraining
        if 'numTraining' not in agentOpts:
            agentOpts['numTraining'] = options.numTraining
    hero = heroType(**agentOpts)  # Instantiate Hero with agentArgs
    args['hero'] = hero

    # Don't display training games
    if 'numTrain' in agentOpts:
        options.numQuiet = int(agentOpts['numTrain'])
        options.numIgnore = int(agentOpts['numTrain'])

    # Choose a dragon agent
    dragonType = loadAgent(options.dragon, noKeyboard)
    args['dragons'] = [dragonType(i+1) for i in range(options.numDragons)]

    # Choose a display format
    if options.quietGraphics or (not animate and not options.textGraphics):
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif options.textGraphics:
        import textDisplay
        textDisplay.SLEEP_TIME = options.frameTime
        args['display'] = textDisplay.HeroGraphics()
    else:
        import graphicsDisplay
        args['display'] = graphicsDisplay.HeroGraphics(
            options.zoom, frameTime=options.frameTime)
    args['numGames'] = options.numGames
    args['record'] = options.record
    args['catchExceptions'] = options.catchExceptions
    args['timeout'] = options.timeout

    # Special case: recorded games don't use the runGames method or args structure
    if options.gameToReplay != None:
        print('Replaying recorded game %s.' % options.gameToReplay)
        import pickle
        f = open(options.gameToReplay)
        try:
            recorded = pickle.load(f)
        finally:
            f.close()
        recorded['display'] = args['display']
        replayGame(**recorded)
        sys.exit(0)

    return args


def loadAgent(hero, nographics):
    # Looks through all pythonPath Directories for the right module,
    pythonPathStr = os.path.expandvars("$PYTHONPATH")
    if pythonPathStr.find(';') == -1:
        pythonPathDirs = pythonPathStr.split(':')
    else:
        pythonPathDirs = pythonPathStr.split(';')
    pythonPathDirs.append('.')

    for moduleDir in pythonPathDirs:
        if not os.path.isdir(moduleDir):
            continue
        moduleNames = [f for f in os.listdir(
            moduleDir) if f.endswith('gents.py')]
        for modulename in moduleNames:
            try:
                module = __import__(modulename[:-3])
            except ImportError:
                continue
            if hero in dir(module):
                if nographics and modulename == 'keyboardAgents.py':
                    raise Exception(
                        'Using the keyboard requires graphics (not text display)')
                return getattr(module, hero)
    raise Exception('The agent ' + hero +
                    ' is not specified in any *Agents.py.')


def replayGame(layout, actions, display):
    import heroAgents
    import dragonAgents
    rules = ClassicGameRules()
    agents = [heroAgents.GreedyAgent()] + [dragonAgents.RandomDragon(i+1)
                                             for i in range(layout.getNumDragons())]
    game = rules.newGame(layout, agents[0], agents[1:], display)
    state = game.state
    display.initialize(state.data)

    for action in actions:
            # Execute the action
        state = state.generateSuccessor(*action)
        # Change the display
        display.update(state.data)
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)

    display.finish()


def runGames(layout, hero, dragons, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
    import __main__
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games = []

    for i in range(numGames):
        beQuiet = i < numTraining
        if beQuiet:
                # Suppress output and graphics
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        game = rules.newGame(layout, hero, dragons,
                             gameDisplay, beQuiet, catchExceptions)
        game.run()
        if not beQuiet:
            games.append(game)

        if record:
            import time
            import pickle
            fname = ('recorded-game-%d' % (i + 1)) + \
                '-'.join([str(t) for t in time.localtime()[1:6]])
            f = file(fname, 'w')
            components = {'layout': layout, 'actions': game.moveHistory}
            pickle.dump(components, f)
            f.close()

    if (numGames-numTraining) > 0:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Win Rate:      %d/%d (%.2f)' %
              (wins.count(True), len(wins), winRate))
        print('Record:       ', ', '.join(
            [['Loss', 'Win'][int(w)] for w in wins]))

    return games


if __name__ == '__main__':
    """
    The main function called when hero.py is run
    from the command line:

    > python hero.py

    See the usage string for more details.

    > python hero.py --help
    """
    args = readCommand(sys.argv[1:])  # Get game components based on input
    runGames(**args)

    # import cProfile
    # cProfile.run("runGames( **args )")
    pass
