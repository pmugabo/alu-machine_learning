#!/usr/bin/env python3
'''
    function def absorbing(P): that
    determines if a markov chain is absorbing
'''


import numpy as np


def absorbing(P):
    '''
    Determines if a markov chain is absorbing
    '''
    # Input validation
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n1, n2 = P.shape
    if n1 != n2:
        return False

    absorbing_states = []
    for i in range(n1):
        if P[i][i] == 1:
            absorbing_states.append(i)

    # Must have at least one absorbing state
    if not absorbing_states:
        return False

    non_absorbing = [
        i for i in range(n1) if i not in absorbing_states
        ]
    if not non_absorbing:
        return True

    # Create reachability matrix through matrix powers
    reachable = P.copy()
    for _ in range(n1-1):
        reachable = np.matmul(reachable, P)

    # Check if each non-absorbing state can
    # reach at least one absorbing state
    for state in non_absorbing:
        can_reach_absorbing = False
        for abs_state in absorbing_states:
            if reachable[state][abs_state] > 0:
                can_reach_absorbing = True
                break
        if not can_reach_absorbing:
            return False

    return True
