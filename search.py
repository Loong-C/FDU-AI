# search.py
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
In search.py, you will implement generic search algorithms which are called by
Hero agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """
    Q1 BFS
    Search the shallowest nodes in the search tree first.
    
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    visited = set()
    fringe = util.Queue()
    fringe.push((start, []))
    while not fringe.isEmpty():
        current_state, actions = fringe.pop()
        if problem.isGoalState(current_state):
            return actions
        if current_state not in visited:
            visited.add(current_state)
            for successor, action, _ in problem.getSuccessors(current_state):
                if successor not in visited:
                    fringe.push((successor, actions + [action]))
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """
    Q2 A*
    Search the node that has the lowest combined cost and heuristic first."""
    if hasattr(heuristic, "choose_from_plateau"):
        return jevAStarSearch(problem, heuristic)
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def jevAStarSearch(problem: SearchProblem, heuristic):
    """A* with Jev used only to order nodes sharing the minimum f value."""
    import heapq
    import itertools

    counter = itertools.count()
    start = problem.getStartState()

    begin_search = getattr(heuristic, "begin_search", None)
    if begin_search is not None:
        begin_search(problem)

    start_h = heuristic(start, problem)
    frontier = [(start_h, next(counter), 0, start, [])]
    best_cost = {start: 0}

    try:
        while frontier:
            first = None
            while frontier and first is None:
                candidate = heapq.heappop(frontier)
                if best_cost.get(candidate[3]) == candidate[2]:
                    first = candidate
            if first is None:
                break

            min_f = first[0]
            plateau = [first]
            while frontier and frontier[0][0] == min_f:
                candidate = heapq.heappop(frontier)
                if best_cost.get(candidate[3]) == candidate[2]:
                    plateau.append(candidate)

            for candidate in plateau:
                if problem.isGoalState(candidate[3]):
                    return candidate[4]

            if len(plateau) == 1:
                selected = plateau[0]
            else:
                selector_candidates = [
                    {
                        "state": state,
                        "g": cost,
                        "h": f_value - cost,
                        "f": f_value,
                        "order": order,
                        "entry": entry,
                    }
                    for entry in plateau
                    for f_value, order, cost, state, path in [entry]
                ]
                selected_candidate = heuristic.choose_from_plateau(
                    selector_candidates,
                    problem,
                )
                selected = selected_candidate["entry"]

            for candidate in plateau:
                if candidate is not selected:
                    heapq.heappush(frontier, candidate)

            _, _, cost, state, path = selected
            for successor, action, step_cost in problem.getSuccessors(state):
                new_cost = cost + step_cost
                if new_cost >= best_cost.get(successor, float("inf")):
                    continue

                best_cost[successor] = new_cost
                successor_h = heuristic(successor, problem)
                heapq.heappush(
                    frontier,
                    (
                        new_cost + successor_h,
                        next(counter),
                        new_cost,
                        successor,
                        path + [action],
                    ),
                )

        return []
    finally:
        statistics = getattr(heuristic, "statistics", None)
        if statistics is not None:
            stats = statistics()
            print(
                "[Jev] requests=%d scored=%d cache_hits=%d "
                "decisions=%d api_time=%.3fs failures=%d retries=%d "
                "throttle_time=%.3fs"
                % (
                    stats["api_requests"],
                    stats["scored_states"],
                    stats["cache_hits"],
                    stats["decision_rounds"],
                    stats["api_seconds"],
                    stats["failures"],
                    stats.get("retries", 0),
                    stats.get("throttle_seconds", 0.0),
                )
            )
        close = getattr(heuristic, "close", None)
        if close is not None:
            close()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
