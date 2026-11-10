"""Q6 feature selection/design.

Choose an existing extractor, combine several, or implement getFeatures yourself.
The initial BiasExtractor is deliberately weak.
"""

from Agents.Runtime.feature_extractors import FeatureExtractor, SimpleExtractor
from Game.Core.game import Actions


class StudentFeatureExtractor(FeatureExtractor):
    def __init__(self):
        # YOUR FEATURE DESIGN HERE: choose, combine, or replace this extractor.
        self.extractor = SimpleExtractor()

    def getFeatures(self, state, action):
        """Return finite feature values without modifying the input GameState."""
        # You may replace this delegation with your own feature computation.
        features = self.extractor.getFeatures(state, action)
        walls = state.getWalls()
        x, y = state.getHeroPosition()
        dx, dy = Actions.directionToVector(action)
        next_position = (int(x + dx), int(y + dy))
        takes_sword = next_position in state.getSwords()

        # Count active dragons and those whose weakened timer expires next turn.
        danger = 0
        for dragon in state.getDragonStates():
            if dragon.weakenedTimer <= 1:
                neighbors = Actions.getLegalNeighbors(dragon.getPosition(), walls)
                if next_position in neighbors:
                    danger += 1
        if takes_sword:
            danger = 0

        # SimpleExtractor already divides its features by 10.
        features['#-of-dragons-1-step-away'] = danger / 10.0
        features['collects-coin'] = float(
            danger == 0 and state.getCoins()[next_position[0]][next_position[1]]) / 10.0
        features['takes-sword'] = float(takes_sword) / 10.0
        return features
