# layout.py
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


import os
from pathlib import Path
from Game.Core.layout import Layout

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def getLayout(name, back=2):
    """Load a layout without changing the process working directory."""
    filename = name if name.endswith('.lay') else name + '.lay'
    roots = [os.path.abspath('.')]
    for _ in range(max(0, back + 1)):
        roots.append(os.path.dirname(roots[-1]))
    roots.append(str(PROJECT_ROOT))
    for root in dict.fromkeys(roots):
        for candidate in [os.path.join(root, 'Assets', 'Layouts', filename), os.path.join(root, filename)]:
            result = tryToLoad(candidate)
            if result is not None:
                return result
    return None


def tryToLoad(fullname):
    if not os.path.isfile(fullname):
        return None
    with open(fullname, encoding='utf-8') as handle:
        return Layout([line.rstrip('\r\n') for line in handle])
