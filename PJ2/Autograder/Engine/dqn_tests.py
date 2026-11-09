"""Behavioral tests for the Q7 student components; PyTorch is not required."""

import math
import random

from Agents.Runtime.module_loader import load_module_file
from Autograder.Engine import test_classes


class DQNComponentTest(test_classes.TestCase):
    def execute(self, grades, moduleDict, solutionDict):
        source = moduleDict['studentDqnUtils']
        try:
            if source.error or not source.path:
                raise FileNotFoundError(source.error or 'Student DQN module is missing')
            module = load_module_file('Agents.Coursework.dqn_utils', source.path)
            self.check(module)
        except (Exception, SystemExit) as error:
            self.addMessage('%s: %s' % (type(error).__name__, error))
            return self.testFail(grades)
        return self.testPass(grades)

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w', encoding='utf-8') as handle:
            handle.write('# This test checks behavior directly.\n')
        return True


class EpsilonSchedulerTest(DQNComponentTest):
    def check(self, module):
        for start, end, duration in ((1.0, 0.05, 100), (0.7, 0.2, 7), (0.3, 0.3, 10)):
            scheduler = module.EpsilonScheduler(start, end, duration)
            for step in (-2, 0, 1, duration // 2, duration, duration + 25):
                fraction = min(1.0, max(0.0, step / duration))
                expected = start + (end - start) * fraction
                actual = scheduler.value(step)
                if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                    raise AssertionError('value(%s) returned %s; expected %s' % (step, actual, expected))


class ReplayBufferTest(DQNComponentTest):
    def check(self, module):
        buffer = module.ReplayBuffer(3)
        if len(buffer) != 0:
            raise AssertionError('A new buffer must be empty')
        records = [(i, 'action', float(i), None if i == 2 else i + 1, (True, False))
                   for i in range(5)]
        for i, record in enumerate(records):
            buffer.add(record)
            if len(buffer) != min(i + 1, 3):
                raise AssertionError('Buffer length or capacity is incorrect')
        batch = buffer.sample(3)
        if not isinstance(batch, list) or len(batch) != 3:
            raise AssertionError('sample must return a list of batch_size transitions')
        if {item[0] for item in batch} != {2, 3, 4}:
            raise AssertionError('The oldest transitions must be discarded and samples must be distinct')
        if any(item != records[item[0]] for item in batch) or len(buffer) != 3:
            raise AssertionError('Sampling must preserve transitions and not remove them')

        # A generous deterministic frequency check rejects biased or fixed-order sampling.
        saved_state = random.getstate()
        try:
            random.seed(41723)
            counts = {2: 0, 3: 0, 4: 0}
            for _ in range(1200):
                result = buffer.sample(1)
                if not isinstance(result, list) or len(result) != 1 or result[0][0] not in counts:
                    raise AssertionError('Single-item samples must contain one stored transition')
                counts[result[0][0]] += 1
            if any(not 300 <= count <= 500 for count in counts.values()):
                raise AssertionError('Sampling should be approximately uniform: %s' % counts)
            if len(buffer) != 3:
                raise AssertionError('Sampling must not remove transitions')
        finally:
            random.setstate(saved_state)
