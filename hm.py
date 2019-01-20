#!/bin/python
import numpy as np
from itertools import product
import random

GIDX = 0
at = 3
max_relevance = 1
#Bin labels include the first and exclude the last.
bin_labels = [[0.05, 0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.8,0.9], [0.9,0.95]]

def relevance_combinations(relevances, _at=0):
    for rel in range(max_relevance + 1):
        relevances[_at] = rel
        if _at + 1 == len(relevances):
            yield relevances
        else:
            yield from relevance_combinations(relevances, _at+1)

def generate_all_pairs(size=at):
    for relevances_A in relevance_combinations(np.zeros(at)):
        for relevances_B in relevance_combinations(np.zeros(at)):
            yield (np.array(relevances_A), np.array(relevances_B))

def get_prob_from_relevance(relevance, max_relevance=1):
    return (np.power(2, relevance) - 1) / 2**max_relevance

def calculate_ERR(relevances):
    thetas = get_prob_from_relevance(relevances)
    sum_res = 0
    for r in range(at):
        prod = 1
        for i in range(r):
            prod *= (1-thetas[i])
        prod *= thetas[r]
        sum_res += prod * (1/(r+1))
    return sum_res

def find_bin_index(delta_err):
    for i, bini in enumerate(bin_labels):
        if delta_err >= bini[0] and delta_err < bini[1]:
            return i
    return -1

def generate_same_documents_sets(relevances_A, relevances_B, same_as=([None, None, None], [None, None, None]), _at=0):
    if _at == len(relevances_A):
        yield same_as
        return
    yield from generate_same_documents_sets(relevances_A, relevances_B, same_as=same_as, _at=_at+1)
    for idx, relB in enumerate(relevances_B):
        if relevances_A[_at] == relB and idx not in same_as[0]:
            same_as[0][_at] = idx
            same_as[1][idx] = _at
            yield from generate_same_documents_sets(relevances_A, relevances_B, same_as=same_as, _at=_at+1)
            same_as[0][_at] = None
            same_as[1][idx] = None

def prob_softmax(at, tau = 3):
    ranks = 1/((np.array(range(at)) + 1)**tau)
    return ranks

def generate_interleaving(relevances_A, relevances_B, initDist, chooseFunction, k=50):
    relevances_AB = (relevances_A, relevances_B)
    for same_as, _ in product(generate_same_documents_sets(relevances_A, relevances_B), range(k)):
        probDistr = initDist.copy()
        relevances, attribution = [], []
        while np.sum(probDistr) > 0:
            R = random.randint(0,1)
            if probDistr[R].sum() == 0:
                continue
            Didx = chooseFunction(probDistr, R)
            probDistr[R][Didx] = 0
            if same_as[R][Didx]:
                probDistr[(R+1)%2][same_as[R][Didx]] = 0
            relevances.append(relevances_AB[R][Didx])
            attribution.append(R)
        yield relevances, attribution

def generate_team_draft_interleavings(relevances_A, relevances_B, k=50):
    initDist = np.ones((2,at),bool)
    chooseFunction = lambda probDistr, R: probDistr[R].argmax()
    yield from generate_interleaving(relevances_A, relevances_B, initDist, chooseFunction)

def generate_probabilistic_interleavings(relevances_A, relevances_B):
    initDist = np.array([prob_softmax(at), prob_softmax(at)])
    chooseFunction = lambda probDistr, R: np.random.choice(at, p=probDistr[R]/probDistr[R].sum())
    yield from generate_interleaving(relevances_A, relevances_B, initDist, chooseFunction)

def generate_marginalized_probabilistic_interleavings(relevances_A, relevances_B):
    pass

def generate_DERR_bins():
    filled_bins = [[] for i in range(len(bin_labels))] # list of [(relevance_E, relevance_P), ...]. Each list corresponds to the bin with limits in bin_labels
    for relevances_E, relevances_P in generate_all_pairs():
        err_E = calculate_ERR(relevances_E)
        err_P = calculate_ERR(relevances_P)
        delta_err = err_E - err_P
        bIdx = find_bin_index(delta_err)
        if bIdx != -1:
            filled_bins[bIdx].append((relevances_E, relevances_P))
    return filled_bins

def main():
    bins = generate_DERR_bins()
    I = 0
    for bIdx, _bin in enumerate(bins):
        print(bin_labels[bIdx])
        for pair in _bin:
            for relevances, attribution in generate_team_draft_interleavings(pair[0], pair[1]):
                I += 1
                pass
                # print(relevances)
    print(I)


if __name__ == "__main__":
    main()
