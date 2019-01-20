import numpy as np

def next_combination(relevances):
    for i in range(at):
        if relevances[i] != max_relevance:
            relevances[i] += 1
            break
        else:
            relevances[i] = 0
            if i == at-1:
                return False
    return True

def next_pair_combination(relevances_A, relevances_B):
    if not next_combination(relevances_A):
        relevances_A = [0,0,0,0,0]
        if not next_combination(relevances_B):
            return False
    return True

