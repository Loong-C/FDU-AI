"""Q6 feature selection/design.

Choose an existing extractor, combine several, or implement getFeatures yourself.
The initial BiasExtractor is deliberately weak.
"""

from Agents.Runtime.feature_extractors import (
    FeatureExtractor, BiasExtractor, CoinExtractor, DragonExtractor,
    CombinedExtractor, IdentityExtractor, CoordinateExtractor,
)


class StudentFeatureExtractor(FeatureExtractor):
    def __init__(self):
        # YOUR FEATURE DESIGN HERE: choose, combine, or replace this extractor.
        self.extractor = BiasExtractor()

    def getFeatures(self, state, action):
        """Return finite feature values without modifying the input GameState."""
        # You may replace this delegation with your own feature computation.
        return self.extractor.getFeatures(state, action)
