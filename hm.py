#!/bin/python
import math
import numpy as np
from itertools import product
import random
import scipy.stats
import json
from ClickModel import BinaryRelevancePBM

AT = 3
MAXREL = 1
#Bin labels include the first and exclude the last.
BIN_LABELS = [[0.05, 0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.8,0.9], [0.9,0.95]]

def relevance_combinations(relevances, at=0):
    for rel in range(MAXREL + 1):
        relevances[at] = rel
        if at + 1 == len(relevances):
            yield relevances
        else:
            yield from relevance_combinations(relevances, at+1)

def generate_all_pairs(size=AT):
    for rel_left in relevance_combinations(np.zeros(size)):
        for rel_right in relevance_combinations(np.zeros(size)):
            yield (np.array(rel_left), np.array(rel_right))

def get_prob_from_relevance(relevance, max_relevance=MAXREL):
    return (np.power(2, relevance) - 1) / 2**max_relevance

def calculate_ERR(relevances, at=AT):
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
    for i, bini in enumerate(BIN_LABELS):
        if delta_err >= bini[0] and delta_err < bini[1]:
            return i
    return -1

def generate_same_documents_sets(rel_pair, same_as=([None, None, None], [None, None, None]), at=0):
    if at == len(rel_pair[0]):
        yield same_as
        return
    yield from generate_same_documents_sets(rel_pair, same_as=same_as, at=at+1)
    for idx, rel_r in enumerate(rel_pair[1]):
        if rel_pair[0][at] == rel_r and idx not in same_as[0]:
            same_as[0][at] = idx
            same_as[1][idx] = at
            yield from generate_same_documents_sets(rel_pair, same_as=same_as, at=at+1)
            same_as[0][at] = None
            same_as[1][idx] = None

def prob_softmax(at, tau = 3):
    ranks = 1/((np.array(range(at)) + 1)**tau)
    return ranks

def generate_interleaving(rel_pair, initDist, chooseFunction, k=600, at=AT):
    for same_as, _ in product(generate_same_documents_sets(rel_pair), range(k)):
        probDistr = initDist.copy()
        relevances, attribution = np.zeros(at), np.zeros(at)
        for i in range(at):
            R = random.randint(0,1)
            if probDistr[R].sum() == 0:
                continue
            Didx = chooseFunction(probDistr, R)
            probDistr[R][Didx] = 0
            if same_as[R][Didx]:
                probDistr[(R+1)%2][same_as[R][Didx]] = 0
            relevances[i] = rel_pair[R][Didx]
            attribution[i] = R
        yield relevances, attribution

def generate_team_draft_interleavings(rel_pair, k=50, at=AT):
    initDist = np.ones((2,at),bool)
    chooseFunction = lambda probDistr, R: probDistr[R].argmax()
    yield from generate_interleaving(rel_pair, initDist, chooseFunction)

def generate_probabilistic_interleavings(rel_pair, at=AT):
    initDist = np.array([prob_softmax(at), prob_softmax(at)])
    chooseFunction = lambda probDistr, R: np.random.choice(at, p=probDistr[R]/probDistr[R].sum())
    yield from generate_interleaving(rel_pair, initDist, chooseFunction)

def generate_DERR_bins():
    filled_bins = [[] for _ in range(len(BIN_LABELS))] # list of [(relevance_E, relevance_P), ...]. Each list corresponds to the bin with limits in bin_labels
    for relevances_E, relevances_P in generate_all_pairs():
        err_E = calculate_ERR(relevances_E)
        err_P = calculate_ERR(relevances_P)
        delta_err = err_E - err_P
        bIdx = find_bin_index(delta_err)
        if bIdx != -1:
            filled_bins[bIdx].append((relevances_E, relevances_P))
    return filled_bins

def click(relevances):
    return random.randint(0,2)

def victory_proportion(rel_pair, click_model, n_impressions_per_interleaving=50, interleaving_generation=generate_team_draft_interleavings):
    victories = [0,0]
    for relevances, attribution in interleaving_generation(rel_pair):
        #clicks are a list of booleans that represent whether that rank was clicked or not
        pair_wins_E = 0
        pair_wins_P = 0
        for _ in range(n_impressions_per_interleaving):
            clicks = click_model.get_clicks(relevances) #
            clicksE = sum(click for i, click in enumerate(clicks) if attribution[i]==0)
            clicksP = sum(click for i, click in enumerate(clicks) if attribution[i]==1)
            if clicksE > clicksP:
                victories[0] += 1
                pair_wins_E += 1
            elif clicksP > clicksE:
                victories[1] += 1
                pair_wins_P += 1
        # if pair_wins_E / (pair_wins_E + pair_wins_P) < 0.51:
        #     print(rel_pair, pair_wins_E / (pair_wins_E + pair_wins_P), relevances, attribution)
    return victories[0]/(victories[0] + victories[1])

def estimate_sample_size(p1, alpha=0.05, beta=0.90, p0=0.5):
    z_alpha = scipy.stats.norm.ppf(1-alpha)
    z_beta = scipy.stats.norm.ppf(beta)
    nDash = ((z_alpha * math.sqrt(p0 * (1-p0)) + z_beta * math.sqrt(p1 * (1-p1))) / abs(p1 - p0))**2
    n = nDash + (1 / abs(p1 - p0))
    return n

def main():
    pbm = BinaryRelevancePBM(file_p_examinations="p_examinations.json")
    bins = generate_DERR_bins()
    for bIdx, _bin in enumerate(bins):
        if len(_bin) == 0:
            print('Empty bin: {}'.format(BIN_LABELS[bIdx]))
            continue
        Ns = []
        pVictorys = []
        for rel_pair in _bin:
            pVictory = victory_proportion(rel_pair, pbm, interleaving_generation=generate_probabilistic_interleavings)
            n = estimate_sample_size(pVictory)
            Ns.append(n)
            pVictorys.append(pVictory)
        Ns = np.array(Ns)
        pVictorys = np.array(pVictorys)
        for i in range(len(Ns)):
            print(_bin[i], Ns[i], pVictorys[i])
        print('{},  MeanPWins: {}, Min: {},    Mean: {},   Max: {}'.format(
            BIN_LABELS[bIdx],
            pVictorys.mean(),
            Ns.min(),
            Ns.mean(),
            Ns.max()))


if __name__ == "__main__":
    main()
