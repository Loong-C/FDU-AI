"""Q6 feature effect scoring; numerical tests remain in reinforcement_test_classes."""

from Autograder.Engine.test_classes import TestCase
from Game.Application.feature_evaluation import (
    DISCOUNT, LAYOUTS, MAX_SECONDS, REPEATS, TRAINING_GAMES, TOTAL_TEST_GAMES,
    points_for_wins, run_isolated, validate_result,
)


class FeaturePerformanceTest(TestCase):
    def execute(self, grades, moduleDict, solutionDict):
        grades.addMessage('Q6 feature experiment: gamma=%.2f; %d training + %d test games; total limit %ds.' %
                          (DISCOUNT, len(LAYOUTS) * REPEATS * TRAINING_GAMES,
                           TOTAL_TEST_GAMES, MAX_SECONDS))
        remaining = grades.remainingTime()
        if remaining <= 5:
            grades.addMessage('Overall grading time budget exhausted; numerical base credit retained.')
            return True
        result = run_isolated(moduleDict['qlearningAgents'].__file__,
                              moduleDict.get('studentFeatureExtractors'),
                              timeout=min(MAX_SECONDS, remaining - 5))
        if not result.get('ok'):
            grades.addMessage('Feature stage: 0/3. ' + result.get('error', 'Evaluation failed'))
            grades.addMessage('Numerical base credit is unchanged.')
            if result.get('log'):
                grades.addMessage(result['log'][-2000:])
            return True
        result = validate_result(result)
        for name in LAYOUTS:
            groups = [g for g in result['groups'] if g['layout'] == name]
            grades.addMessage('%s: %d/%d wins; mean score %.2f; coin fraction %.3f; %d truncated.' % (
                name, sum(g['wins'] for g in groups), sum(g['tests'] for g in groups),
                sum(g['mean_score'] for g in groups) / len(groups),
                sum(g['coin_fraction'] for g in groups) / len(groups),
                sum(g['truncated'] for g in groups)))
        points = points_for_wins(result['wins'])
        grades.addPoints(points)
        grades.addMessage('Feature stage: %d/3; %d/%d wins (%.2f%%); %.2fs.' % (
            points, result['wins'], result['tests'], 100 * result['win_rate'], result['seconds']))
        return True

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# Performance is measured by the fixed experiment, not reference weight values.\n')
        return True
