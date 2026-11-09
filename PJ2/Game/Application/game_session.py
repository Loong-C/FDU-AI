# game.py
# -------
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


# game.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The original AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://ai.berkeley.edu.

import sys, time, traceback
from Game.Application.timeout import TimeoutFunction, TimeoutFunctionException
try:
    import boinc
    _BOINC_ENABLED = True
except:
    _BOINC_ENABLED = False

class Game:
    """
    The Game manages the control flow, soliciting actions from agents.
    """

    def __init__(self, agents, horizon, display, rules, startingIndex=0, muteAgents=False, catchExceptions=False,
                 observationFactory=None):
        self.agentCrashed = False
        self.agents = agents
        self.display = display
        self.rules = rules
        self.startingIndex = startingIndex
        self.gameOver = False
        self.muteAgents = muteAgents
        self.catchExceptions = catchExceptions
        self._observationFactory = observationFactory
        self.moveHistory = []
        self.totalAgentTimes = [0 for agent in agents]
        self.totalAgentTimeWarnings = [0 for agent in agents]
        self.agentTimeout = False
        self.horizon = horizon
        self.truncated = False
        self.started = False
        self._display_started = False
        self._display_finished = False
        self._previousStreams = None
        import io
        self.agentOutput = [io.StringIO() for agent in agents]

    def report(self, message):
        print(message)

    def _observation(self):
        if self._observationFactory is not None:
            return self._observationFactory(self.state)
        return self.state.deepCopy()

    def getProgress(self):
        if self.gameOver:
            return 1.0
        else:
            return self.rules.getProgress(self)

    def _agentCrash(self, agentIndex, quiet=False):
        "Helper method for handling agent crashes"
        if not quiet:
            traceback.print_exc()
        self.gameOver = True
        self.agentCrashed = True
        self.rules.agentCrash(self, agentIndex)

    OLD_STDOUT = None
    OLD_STDERR = None

    def mute(self, agentIndex):
        if not self.muteAgents or self._previousStreams is not None:
            return
        self._previousStreams = (sys.stdout, sys.stderr)
        sys.stdout = self.agentOutput[agentIndex]
        sys.stderr = self.agentOutput[agentIndex]

    def unmute(self):
        if self._previousStreams is None:
            return
        # Revert stdout/stderr to originals
        sys.stdout, sys.stderr = self._previousStreams
        self._previousStreams = None

    def start(self):
        """Initialize the display and agent callbacks before the first tick."""
        if self._display_started:
            raise RuntimeError('Game has already started')
        self._display_started = True
        self.display.initialize(self.state.data)
        self.numMoves = 0
        self.numTicks = 0

        # self.display.initialize(self.state.makeObservation(1).data)
        # inform learning agents of the game start
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                self.mute(i)
                # this is a null agent, meaning it failed to load
                # the other team wins
                print("Agent %d failed to load" % i, file=sys.stderr)
                self.unmute()
                self._agentCrash(i, quiet=True)
                return False
            if ("registerInitialState" in dir(agent)):
                self.mute(i)
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(
                            agent.registerInitialState, self.rules.getMaxStartupTime(i))
                        try:
                            start_time = time.time()
                            timed_func(self._observation())
                            time_taken = time.time() - start_time
                            self.totalAgentTimes[i] += time_taken
                        except TimeoutFunctionException:
                            print("Agent %d ran out of time on startup!" %
                                  i, file=sys.stderr)
                            self.unmute()
                            self.agentTimeout = True
                            self._agentCrash(i, quiet=True)
                            return False
                    except Exception as data:
                        self._agentCrash(i, quiet=False)
                        self.unmute()
                        return False
                else:
                    agent.registerInitialState(self._observation())
                # TODO: could this exceed the total time
                self.unmute()

        self.started = True
        return True

    def _run_ticks(self):
        if not self.started:
            raise RuntimeError('Game must be started before running ticks')
        agentIndex = self.startingIndex
        numAgents = len(self.agents)
        hero_moves = 0

        def within_horizon():
            return (self.horizon < 0 or hero_moves < self.horizon
                    or (hero_moves > 0 and agentIndex != 0))

        while not self.gameOver and within_horizon():
            # Fetch the next agent
            agent = self.agents[agentIndex]
            move_time = 0
            skip_action = False
            # Generate an observation of the state
            if 'observationFunction' in dir(agent):
                self.mute(agentIndex)
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(agent.observationFunction,
                            self.rules.getMoveTimeout(agentIndex))
                        try:
                            start_time = time.time()
                            observation = timed_func(self._observation())
                        except TimeoutFunctionException:
                            skip_action = True
                        move_time += time.time() - start_time
                        self.unmute()
                    except Exception as data:
                        self._agentCrash(agentIndex, quiet=False)
                        self.unmute()
                        return
                else:
                    observation = agent.observationFunction(
                        self._observation())
                self.unmute()
            else:
                observation = self._observation()

            # Solicit an action
            action = None
            self.mute(agentIndex)
            if self.catchExceptions:
                try:
                    timed_func = TimeoutFunction(agent.getAction,
                        self.rules.getMoveTimeout(agentIndex) - move_time)
                    try:
                        start_time = time.time()
                        if skip_action:
                            raise TimeoutFunctionException()
                        action = timed_func(observation)
                    except TimeoutFunctionException:
                        print("Agent %d timed out on a single move!" %
                              agentIndex, file=sys.stderr)
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return

                    move_time += time.time() - start_time

                    if move_time > self.rules.getMoveWarningTime(agentIndex):
                        self.totalAgentTimeWarnings[agentIndex] += 1
                        print("Agent %d took too long to make a move! This is warning %d" % (
                            agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
                        if self.totalAgentTimeWarnings[agentIndex] > self.rules.getMaxTimeWarnings(agentIndex):
                            print("Agent %d exceeded the maximum number of warnings: %d" % (
                                agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
                            self.agentTimeout = True
                            self._agentCrash(agentIndex, quiet=True)
                            self.unmute()
                            return

                    self.totalAgentTimes[agentIndex] += move_time
                    # print "Agent: %d, time: %f, total: %f" % (agentIndex, move_time, self.totalAgentTimes[agentIndex])
                    if self.totalAgentTimes[agentIndex] > self.rules.getMaxTotalTime(agentIndex):
                        print("Agent %d ran out of time! (time: %1.2f)" % (
                            agentIndex, self.totalAgentTimes[agentIndex]), file=sys.stderr)
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return
                    self.unmute()
                except Exception as data:
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                action = agent.getAction(observation)
            self.unmute()

            # Execute the action
            if self.catchExceptions:
                try:
                    self.state = self.state.generateSuccessor(
                        agentIndex, action)
                except Exception as data:
                    self.mute(agentIndex)
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                self.state = self.state.generateSuccessor(agentIndex, action)
            self.moveHistory.append((agentIndex, action))
            if agentIndex == 0:
                hero_moves += 1

            # Change the display
            self.display.update(self.state.data)
            ###idx = agentIndex - agentIndex % 2 + 1
            ###self.display.update( self.state.makeObservation(idx).data )

            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)
            # Track progress
            if agentIndex == (self.startingIndex + numAgents - 1) % numAgents:
                self.numMoves += 1
            # Next agent
            agentIndex = (agentIndex + 1) % numAgents

            if _BOINC_ENABLED:
                boinc.set_fraction_done(self.getProgress())

            if agentIndex == self.startingIndex:
                self.numTicks += 1
                if not self.gameOver and within_horizon():
                    yield

        self.truncated = not self.gameOver
        self.gameOver = True
        # A horizon cut-off keeps the nonterminal state for TD bootstrapping.
        # inform a learning agent of the game result
        for agentIndex, agent in enumerate(self.agents):
            if "final" in dir(agent):
                try:
                    self.mute(agentIndex)
                    agent.final(self._observation())
                    self.unmute()
                except Exception as data:
                    if not self.catchExceptions:
                        raise
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return

    def runTicks(self):
        """Yield after complete rounds, preserving PJ2 action and horizon semantics."""
        try:
            yield from self._run_ticks()
        finally:
            self.finish()

    def finish(self):
        """Restore streams and release display resources exactly once."""
        self.unmute()
        if self._display_started and not self._display_finished:
            self._display_finished = True
            self.display.finish()

    def run(self):
        """Drain ticks without adding real-time scheduling delays."""
        try:
            if self.start():
                for _ in self.runTicks():
                    pass
        finally:
            self.finish()
