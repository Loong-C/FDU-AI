actions = mdp.getPossibleActions(state)
is_terminal = mdp.isTerminal(state)
transitions = mdp.getTransitionStatesAndProbs(state, action)
