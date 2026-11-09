# featureExtractors.py
# --------------------
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


"Feature extractors for Hero game states"

from Game.Core.game import Directions, Actions
from Game.Core import utilities as util

class FeatureExtractor:
    """Return a fresh sparse mapping from hashable keys to finite real values.

    getFeatures must not modify state and must return stable features for the
    same state/action. Q6.2 may cache repeated queries and returns fresh mappings.
    CoordinateExtractor accepts Hero GameState or Gridworld coordinates;
    coin/dragon extractors use GameState and a legal action. Q6.2 evaluation
    supplies read-only GameState observations with the same query API. Use
    state.deepCopy() or grid.copy() for private mutable work. util.Counter is
    the recommended return type.
    """
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    """Four unit indicators for the current position, x, y, and chosen action.

    Hero states contribute their current hero position. Gridworld (x, y) inputs
    retain their original feature keys and values. Neither input domain uses 
    extra scaling.
    """
    def getFeatures(self, state, action):
        position = state.getHeroPosition() if hasattr(state, 'getHeroPosition') else state
        feats = util.Counter()
        feats[position] = 1.0
        feats['x=%d' % position[0]] = 1.0
        feats['y=%d' % position[1]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestCoin(pos, coins, walls):
    """
    closestCoin -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    layers = getattr(walls, '_closest_coin_distance', None)
    if (layers is not None and 0 <= pos[0] < walls.width and 0 <= pos[1] < walls.height
            and not walls[pos[0]][pos[1]]):
        return layers(pos, coins)
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a coin at this location then exit
        if coins[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no coins found
    return None

class BiasExtractor(FeatureExtractor):
    """Hero weak baseline: one constant feature, scaled by 0.1."""
    def getFeatures(self, state, action):
        return util.Counter({'bias': 0.1})

class CoinExtractor(FeatureExtractor):
    """Hero coin features."""
    def getFeatures(self, state, action):
        coins, walls = state.getCoins(), state.getWalls()
        x, y = state.getHeroPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        features = util.Counter({'bias': 1.0,
                                 'collects-coin': float(coins[next_x][next_y])})
        distance = closestCoin((next_x, next_y), coins, walls)
        if distance is not None:
            features['closest-coin'] = float(distance) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class DragonExtractor(FeatureExtractor):
    """Hero one-step dragon danger features."""
    def getFeatures(self, state, action):
        x, y = state.getHeroPosition()
        dx, dy = Actions.directionToVector(action)
        position = int(x + dx), int(y + dy)
        walls = state.getWalls()
        danger = sum(position in Actions.getLegalNeighbors(g, walls)
                     for g in state.getDragonPositions())
        features = util.Counter({'bias': 1.0, '#-of-dragons-1-step-away': danger})
        features.divideAll(10.0)
        return features

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Hero:
    - whether a coin will be collected
    - how far away the next coin is
    - whether a dragon collision is imminent
    - whether a dragon is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of coins and wall locations and get the dragon locations
        coins = state.getCoins()
        walls = state.getWalls()
        dragons = state.getDragonPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of hero after he takes the action
        x, y = state.getHeroPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of dragons 1-step away
        features["#-of-dragons-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in dragons)

        # if there is no danger of dragons then add the coins feature
        if not features["#-of-dragons-1-step-away"] and coins[next_x][next_y]:
            features["collects-coin"] = 1.0

        dist = closestCoin((next_x, next_y), coins, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-coin"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class CombinedExtractor(FeatureExtractor):
    """Combine instances without extra scaling; keys become (index, original_key)."""
    def __init__(self, extractors):
        self.extractors = tuple(extractors)

    def getFeatures(self, state, action):
        features = util.Counter()
        for index, extractor in enumerate(self.extractors):
            for key, value in extractor.getFeatures(state, action).items():
                features[(index, key)] = value
        return features
