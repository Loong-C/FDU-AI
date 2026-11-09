"""Read-only, value-compatible observations for long learning experiments.

The engine keeps ordinary GameState objects. These snapshots expose the same
query API without sharing writable grids, configurations or agent data. Static
layout and unchanged coin grids are reused by an observation factory. This is an
interface contract, not a sandbox for arbitrary Python code.
"""

from types import MappingProxyType

from Game.Core.game import AgentState, Configuration, Grid, Actions
from Game.Core.layout import Layout
from Game.Core.rules import GameState
from Game.Core.game import GameStateData


class _ReadOnly:
    def __setattr__(self, name, value):
        raise TypeError('GameState observations are read-only; copy before editing')

    def __delattr__(self, name):
        raise TypeError('GameState observations are read-only')


def _set(instance, **values):
    for name, value in values.items():
        object.__setattr__(instance, name, value)


class ReadOnlyGrid(_ReadOnly, Grid):
    def __init__(self, grid):
        columns = tuple(tuple(column) for column in grid.data)
        _set(self, width=grid.width, height=grid.height,
             CELLS_PER_INT=grid.CELLS_PER_INT, data=columns,
             _true_count=sum(column.count(True) for column in columns),
             _false_count=sum(column.count(False) for column in columns),
             _cached_hash=None, _neighbors=MappingProxyType({}),
             _distance_layers=MappingProxyType({}))

    def __setitem__(self, key, value):
        raise TypeError('GameState grids are read-only; use grid.copy() for a private copy')

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Grid):
            return False
        return self.data == (other.data if isinstance(other, ReadOnlyGrid)
                             else tuple(tuple(column) for column in other.data))

    def __hash__(self):
        if self._cached_hash is None:
            object.__setattr__(self, '_cached_hash', Grid.__hash__(self))
        return self._cached_hash

    def count(self, item=True):
        if item == True:
            return self._true_count
        if item == False:
            return self._false_count
        return sum(column.count(item) for column in self.data)

    def copy(self):
        grid = Grid(self.width, self.height)
        grid.data = [list(column) for column in self.data]
        return grid

    deepCopy = copy
    shallowCopy = copy

    def _legal_neighbors(self, x, y):
        # Cache immutable topology only, never student features or Q values.
        key = x, y
        if key not in self._neighbors:
            values = []
            for _, (dx, dy) in Actions._directionsAsList:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx == self.width or ny < 0 or ny == self.height:
                    continue
                if not self[nx][ny]:
                    values.append((nx, ny))
            updated = dict(self._neighbors)
            updated[key] = tuple(values)
            object.__setattr__(self, '_neighbors', MappingProxyType(updated))
        return list(self._neighbors[key])

    def _closest_coin_distance(self, position, coins):
        # Static shortest-distance layers. Coin contents are read anew on every
        # query; neither changing coins nor student feature values are cached.
        position = tuple(position)
        if position not in self._distance_layers:
            visited = {position}
            frontier = (position,)
            layers = []
            while frontier:
                layers.append(frontier)
                following = []
                for x, y in frontier:
                    for neighbor in self._legal_neighbors(x, y):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            following.append(neighbor)
                frontier = tuple(following)
            updated = dict(self._distance_layers)
            updated[position] = tuple(layers)
            object.__setattr__(self, '_distance_layers', MappingProxyType(updated))
        for distance, layer in enumerate(self._distance_layers[position]):
            for x, y in layer:
                if coins[x][y]:
                    return distance
        return None


class _Configuration(_ReadOnly, Configuration):
    def __init__(self, configuration):
        _set(self, pos=tuple(configuration.pos), direction=configuration.direction)


class _AgentState(_ReadOnly, AgentState):
    def __init__(self, agent):
        start = _Configuration(agent.start) if agent.start is not None else None
        configuration = (start if agent.configuration is agent.start else
                         _Configuration(agent.configuration) if agent.configuration is not None else None)
        _set(self, start=start, configuration=configuration, isHero=agent.isHero,
             weakenedTimer=agent.weakenedTimer, numCarrying=agent.numCarrying,
             numReturned=agent.numReturned)


def _visibility(values):
    return tuple(tuple(MappingProxyType({direction: frozenset(positions)
                                        for direction, positions in cell.items()})
                       for cell in column) for column in values)


class _Layout(_ReadOnly, Layout):
    def __init__(self, layout):
        _set(self, width=layout.width, height=layout.height,
             walls=ReadOnlyGrid(layout.walls), coins=ReadOnlyGrid(layout.coins),
             swords=tuple(layout.swords), agentPositions=tuple(layout.agentPositions),
             numDragons=layout.numDragons, totalCoins=layout.totalCoins,
             layoutText=tuple(layout.layoutText))
        if hasattr(layout, 'visibility'):
            object.__setattr__(self, 'visibility', _visibility(layout.visibility))

    def deepCopy(self):
        return Layout(list(self.layoutText))

    def initializeVisibilityMatrix(self):
        if not hasattr(self, 'visibility'):
            layout = self.deepCopy()
            layout.initializeVisibilityMatrix()
            object.__setattr__(self, 'visibility', _visibility(layout.visibility))


class _StateData(_ReadOnly, GameStateData):
    def __init__(self, data, coins, layout):
        _set(self, coins=coins, layout=layout, swords=tuple(data.swords),
             agentStates=tuple(_AgentState(agent) for agent in data.agentStates),
             _defeated=tuple(data._defeated), score=data.score,
             scoreChange=data.scoreChange, _lose=data._lose, _win=data._win,
             _coinCollected=data._coinCollected, _coinAdded=data._coinAdded,
             _swordCollected=data._swordCollected, _agentMoved=data._agentMoved,
             _cached_hash=None)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, GameStateData):
            return False
        return (self.agentStates == tuple(other.agentStates)
                and self.coins == other.coins and self.swords == tuple(other.swords)
                and self.score == other.score and self._win == other._win
                and self._lose == other._lose and self.layout.walls == other.layout.walls)

    def __hash__(self):
        if self._cached_hash is None:
            value = hash((self.agentStates, self.coins, self.swords, self.score,
                          self._win, self._lose, self.layout.walls))
            object.__setattr__(self, '_cached_hash', value)
        return self._cached_hash

    def deepCopy(self):
        data = GameStateData(self)
        data.coins = self.coins.copy()
        data.layout = self.layout.deepCopy()
        data.swords = list(self.swords)
        data._defeated = list(self._defeated)
        for name in ('scoreChange', '_agentMoved', '_coinCollected', '_coinAdded', '_swordCollected'):
            setattr(data, name, getattr(self, name))
        return data


class ReadOnlyGameState(_ReadOnly, GameState):
    """A stable observation with GameState query, equality and hash semantics.

    deepCopy returns a detached mutable GameState, so private simulations and
    feature construction on a copy remain possible. Successor queries return
    new read-only observations and never modify the input or running game.
    """
    def __init__(self, state, coins, layout):
        _set(self, data=_StateData(state.data, coins, layout),
             _trackExplored=getattr(state, '_trackExplored', True),
             _legal_actions=MappingProxyType({}))

    def getLegalActions(self, agentIndex=0):
        if agentIndex not in self._legal_actions:
            actions = tuple(GameState.getLegalActions(self, agentIndex))
            updated = dict(self._legal_actions)
            updated[agentIndex] = actions
            object.__setattr__(self, '_legal_actions', MappingProxyType(updated))
        return list(self._legal_actions[agentIndex])

    def getDragonStates(self):
        return list(self.data.agentStates[1:])

    def getSwords(self):
        # Preserve the list-returning query API without exposing engine storage.
        return list(self.data.swords)

    def deepCopy(self):
        state = GameState()
        state.data = self.data.deepCopy()
        state._trackExplored = self._trackExplored
        return state

    def generateSuccessor(self, agentIndex, action):
        successor = self.deepCopy().generateSuccessor(agentIndex, action)
        return ReadOnlyObservations()(successor)


class ReadOnlyObservations:
    """Factory for an engine that replaces GameState on each transition.

    Coin storage is shared by the engine until a collection copies it; layouts
    are static. Only the latest mutable source is retained, never game histories.
    Returned observations contain no writable references to engine data.
    """
    def __init__(self):
        self._state = self._observation = None
        self._layout_source = self._layout = None
        self._coin_source = self._coins = None

    def __call__(self, state):
        if isinstance(state, ReadOnlyGameState):
            return state
        if state is self._state:
            return self._observation
        if state.data.layout is not self._layout_source:
            self._layout_source = state.data.layout
            self._layout = _Layout(state.data.layout)
        if state.data.coins.data is not self._coin_source:
            self._coin_source = state.data.coins.data
            self._coins = ReadOnlyGrid(state.data.coins)
        self._state = state
        self._observation = ReadOnlyGameState(state, self._coins, self._layout)
        return self._observation
