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


import sys, time, random, os
from Game.Application.game_session import Game
from Game.Core.rules import GameState, ClassicGameRules
from Game.Application import layout_loader as layout

def createGame(rules, layout, horizon, heroAgent, dragonAgents, display, quiet=False, catchExceptions=False,
               observationFactory=None, trackExplored=True):
    if sum(isHero for isHero, _ in layout.agentPositions) != 1:
        raise ValueError('A game layout must contain exactly one Hero (P)')
    agents = [heroAgent] + dragonAgents[:layout.getNumDragons()]
    initState = GameState()
    initState.initialize(layout, len(dragonAgents))
    initState._trackExplored = trackExplored
    game = Game(agents, horizon, display, rules, catchExceptions=catchExceptions,
                observationFactory=observationFactory)
    game.state = initState
    rules.initialState = initState.deepCopy()
    rules.quiet = quiet
    return game


def default(str):
    return str + ' [Default: %default]'


def parseAgentArgs(str):
    if str == None:
        return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=', 1)
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
    USAGE:      python run.py <options>
    EXAMPLES:   (1) python run.py
                    - starts an interactive game
                (2) python run.py --layout smallClassic --zoom 2
                OR  python run.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)
    parser.add_option('-m', dest='maxHorizon', type='int',
                      help=default('The maximum number of Hero actions per game'), metavar='GAMES', default=-1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default(
                          'the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='mediumClassic')
    parser.add_option('-p', '--hero', dest='hero',
                      help=default(
                          'the agent TYPE in Agents.Runtime or Agents.Coursework to use'),
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_option('-t', '--textGraphics', action='store_true', dest='textGraphics',
                      help='Display output as text only', default=False)
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-g', '--dragons', dest='dragon',
                      help=default(
                          'the dragon agent TYPE in Agents.Runtime.dragon_agents to use'),
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
    parser.add_option('--speed', dest='speed', type='float', default=None,
                      help='Game speed multiplier (default: 0.5 for keyboard, 1 otherwise)')
    parser.add_option('-c', '--catchExceptions', action='store_true', dest='catchExceptions',
                      help='Turns on exception handling and timeouts during games', default=False)
    parser.add_option('--timeout', dest='timeout', type='int',
                      help=default('Maximum length of time an agent can spend computing in a single game'), default=30)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    import math
    if options.numGames < 0 or options.numTraining < 0 or options.numDragons < 0:
        parser.error('Game, training and dragon counts must be nonnegative')
    if options.maxHorizon < -1:
        parser.error('The horizon must be -1 (unlimited) or nonnegative')
    if not math.isfinite(options.zoom) or options.zoom <= 0:
        parser.error('Zoom must be positive and finite')
    if not math.isfinite(options.frameTime) or options.timeout <= 0:
        parser.error('Frame time must be finite and timeout must be positive')
    args = dict()

    # Fix the random seed
    if options.fixRandomSeed:
        random.seed('cs188')

    # Choose a layout
    args['layout'] = layout.getLayout(options.layout)
    if args['layout'] == None:
        raise Exception("The layout " + options.layout + " cannot be found")

    args['horizon'] = options.maxHorizon

    # Choose a Hero agent
    noKeyboard = options.gameToReplay == None and (
        options.textGraphics or options.quietGraphics)
    heroType = loadAgent(options.hero, noKeyboard)
    from Agents.Runtime.keyboard_agents import KeyboardAgent
    uses_keyboard = isinstance(heroType, type) and issubclass(heroType, KeyboardAgent)
    speed = options.speed if options.speed is not None else (0.5 if uses_keyboard else 1.0)
    if not math.isfinite(speed) or speed <= 0:
        parser.error('speed must be a positive finite number')
    tick_duration = options.frameTime / speed if options.frameTime >= 0 else None
    if tick_duration is not None and not math.isfinite(tick_duration):
        parser.error('tick duration must be finite')
    if options.hero == "HeroDeepQAgent":
        print("options.agentArgs", options.agentArgs)
        layout_str = "layout_input={}".format(options.layout)
        if options.agentArgs:
            options.agentArgs += ',' + layout_str
        else:
            options.agentArgs = layout_str
    agentOpts = parseAgentArgs(options.agentArgs)
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
    if options.quietGraphics:
        from Game.Presentation import text_display as textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif options.textGraphics:
        from Game.Presentation import text_display as textDisplay
        textDisplay.SLEEP_TIME = options.frameTime
        args['display'] = textDisplay.HeroGraphics()
    else:
        from Game.Presentation import graphics_display as graphicsDisplay
        args['display'] = graphicsDisplay.HeroGraphics(
            options.zoom, frameTime=min(options.frameTime, tick_duration)
            if tick_duration is not None else options.frameTime)
    args['tickDuration'] = tick_duration if uses_keyboard and not options.quietGraphics else None
    args['numGames'] = options.numGames
    args['record'] = options.record
    args['catchExceptions'] = options.catchExceptions
    args['timeout'] = options.timeout

    # Special case: recorded games don't use the runGames method or args structure
    if options.gameToReplay != None:
        print('Replaying recorded game %s.' % options.gameToReplay)
        import pickle
        f = open(options.gameToReplay, 'rb')
        try:
            recorded = pickle.load(f)
        finally:
            f.close()
        recorded['display'] = args['display']
        replayGame(**recorded)
        sys.exit(0)

    return args


def loadAgent(hero, nographics):
    """Find external agents or a canonical bundled Agent module."""
    import importlib
    pythonPathStr = os.path.expandvars('$PYTHONPATH')
    pythonPathDirs = pythonPathStr.split(';' if ';' in pythonPathStr else ':')
    pythonPathDirs.append('.')
    # Preserve explicitly supplied external-agent discovery before built-ins.
    for directory in pythonPathDirs:
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if not filename.endswith('gents.py'):
                continue
            try:
                module = importlib.import_module(filename[:-3])
            except ImportError:
                continue
            if hasattr(module, hero):
                if nographics and filename in {'keyboardAgents.py', 'keyboard_agents.py'}:
                    raise Exception('Using the keyboard requires graphics (not text display)')
                return getattr(module, hero)
    modules = [
        'Agents.Runtime.hero_agents', 'Agents.Runtime.dragon_agents',
        'Agents.Runtime.keyboard_agents', 'Agents.Runtime.learning_agents',
        'Agents.Coursework.qlearning_agents', 'Agents.Coursework.value_iteration_agents',
    ]
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, hero):
            if nographics and name == 'Agents.Runtime.keyboard_agents':
                raise Exception('Using the keyboard requires graphics (not text display)')
            return getattr(module, hero)
    raise Exception('The agent ' + hero + ' is not specified in any Agent module.')


def replayGame(layout, actions, display):
    from Agents.Runtime import hero_agents as heroAgents
    from Agents.Runtime import dragon_agents as dragonAgents
    rules = ClassicGameRules()
    agents = [heroAgents.GreedyAgent()] + [dragonAgents.RandomDragon(i + 1)
                                        for i in range(layout.getNumDragons())]
    game = createGame(rules, layout, -1, agents[0], agents[1:], display)
    try:
        display.initialize(game.state.data)
        for action in actions:
            if game.gameOver:
                raise ValueError('Recorded actions continue after the terminal state')
            game.state = game.state.generateSuccessor(*action)
            game.moveHistory.append(action)
            display.update(game.state.data)
            rules.process(game.state, game)
        return game.state
    finally:
        display.finish()


def runGames(layout, horizon, hero, dragons, display, numGames, record, numTraining=0,
             catchExceptions=False, timeout=30, tickDuration=None, gameRunner=None,
             observationFactory=None, trackExplored=True):
    import __main__
    __main__.__dict__['_display'] = display

    from Game.Application.tick_scheduler import AgentDrivenScheduler, TickScheduler
    if gameRunner is None:
        if tickDuration is None:
            gameRunner = AgentDrivenScheduler().run
        else:
            import math
            if not math.isfinite(tickDuration) or tickDuration < 0:
                raise ValueError('tickDuration must be finite and nonnegative')
            from Game.Presentation import graphics_display, graphics_utils
            wait = graphics_utils.sleep if isinstance(display, graphics_display.HeroGraphics) else time.sleep
            gameRunner = TickScheduler(tickDuration, wait=wait).run

    rules = ClassicGameRules(timeout)
    games = []

    for i in range(numGames):
        # if i % 10 == 0:
        #     print("numGames played: [{}/{}]".format(i, numGames))
        GameState.getAndResetExplored()
        beQuiet = i < numTraining
        if beQuiet:
                # Suppress output and graphics
            from Game.Presentation import text_display as textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        game = createGame(rules, layout, horizon, hero, dragons,
                             gameDisplay, beQuiet, catchExceptions,
                             observationFactory=observationFactory, trackExplored=trackExplored)
        if beQuiet:
            game.run()
        else:
            gameRunner(game)
        if not beQuiet:
            games.append(game)

        if record:
            import pickle
            fname = ('recorded-game-%d' % (i + 1)) + \
                '-'.join([str(t) for t in time.localtime()[1:6]])
            components = {'layout': layout, 'actions': game.moveHistory}
            with open(fname, 'wb') as handle:
                pickle.dump(components, handle)

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
