# multiAgents_student.py
# Student template: copy to multiAgents.py in the distributed pj1-search folder.
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from hero import GameState
from jev.q9 import Q9JevEvaluator
JEV_SCALE = 10
USE_JEV = False
JEV_WEIGHT = 0.10
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (hero.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining coins (newCoins) and Hero position after moving (newPos).
        newWeakenedTimes holds the number of moves that each dragon will remain
        weakened after the Hero has collected a sword.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (hero.py)
        successorGameState = currentGameState.generateHeroSuccessor(action)
        newPos = successorGameState.getHeroPosition()
        newCoins = successorGameState.getCoins()
        newDragonStates = successorGameState.getDragonStates()
        newWeakenedTimes = [dragonState.weakenedTimer for dragonState in newDragonStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Hero GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxHeroAgent, AlphaBetaHeroAgent & ExpectimaxHeroAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Hero is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Hero, dragons are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        agent_nums = gameState.getNumAgents()
        next_agent_index = 1 % agent_nums
        initial_depth = 1 if next_agent_index == 0 else 0
        def value(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            actions = state.getLegalActions(agentIndex)
            successors = [state.generateSuccessor(agentIndex, action) for action in actions]
            successor_values = []
            for successor in successors:
                if agentIndex == agent_nums - 1:
                    successor_values.append(value(successor, depth + 1, 0))
                else:
                    successor_values.append(value(successor, depth, agentIndex + 1))
            if agentIndex == 0:
                return max(successor_values)
            else:
                return min(successor_values)
        actions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in actions]
        best_action = None
        best_value = float('-inf')
        for action, successor in zip(actions, successors):
            v = value(successor, initial_depth, next_agent_index)
            if v > best_value:
                best_value = v
                best_action = action
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def value(state, depth, agentIndex, alpha, beta):
            agent_nums = gameState.getNumAgents()
            next_agent_index = (agentIndex+1) % agent_nums
            next_depth = depth + 1 if next_agent_index == 0 else depth
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            actions = state.getLegalActions(agentIndex)
            if agentIndex == 0:
                current_max =  float('-inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    child_value = value(successor, next_depth, next_agent_index, max(current_max,alpha), beta)
                    if child_value > current_max:
                        current_max = child_value
                    if current_max > beta:
                        return current_max
                return current_max
            if agentIndex > 0:
                current_min = float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    child_value = value(successor, next_depth, next_agent_index, alpha, min(current_min,beta))
                    if child_value < current_min:
                        current_min = child_value
                    if current_min < alpha:
                        return current_min
                return current_min
        agent_nums = gameState.getNumAgents()
        next_agent_index = 1 % agent_nums
        initial_depth = 1 if next_agent_index == 0 else 0
        actions = gameState.getLegalActions(0)
        best_action = None
        best_value = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            v = value(successor, initial_depth, next_agent_index, alpha=best_value, beta=float('inf'))
            if v > best_value:
                best_value = v
                best_action = action
        return best_action
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All dragons should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

# Optional, ungraded extension: reuse Jev to cache repeated states.
_Q9_JEV_EVALUATOR = Q9JevEvaluator()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Q9: design a state evaluation function (higher is better).

    DESCRIPTION: <explain your features, terminal handling and design choices>

    Optional, ungraded Jev extension: jev_score is an overall state-quality
    score in [0, 100] (higher is better), without current game score as input.
    Remote calls may be slow. See the Q9 section in jev/README.md.

    Example:
        jev_score = _Q9_JEV_EVALUATOR.score(currentGameState)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation: keep this name for the autograder.
better = betterEvaluationFunction
