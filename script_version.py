#!/bin/python

import math
import numpy as np
from itertools import product
import scipy.stats
import random
import json
from abc import ABC, abstractmethod
import sys

#How many results to look at
AT = 3

#Maximum relevance of a results
MAXREL = 1

#Bin labels include the first and exclude the last.
BIN_LABELS = [[0.05, 0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7, 0.8], [0.8,0.9], [0.9,0.95]]

class ClickSession:
    #ClickSession is unique for the pair session id and query id
    def __init__(self, _id, query_id, results):
        self._id = _id
        self.query_id = query_id
        self.results = results #List of document ids
        self.clicks_id = []
        self.clicks_rank = [False for i in range(len(results))]

    #The document_id maps to a result of self.results
    def click(self, document_id):
        self.clicks_rank[self.results.index(document_id)] = True
        self.clicks_id.append(document_id)

def parse_log(filename):
    sessions = []
    with open(filename, 'r') as fp:
        last_id = -1
        for line in fp:
            line = line.split()
            if line[2] == "Q":
                _id = line[0]
                last_id = _id
                query_id = line[3]
                results = line[5:]
                sessions.append(ClickSession(_id, query_id, results))
            if line[2] == "C":
                document_id = line[3]
                i = -1
                while sessions[i]._id == last_id and document_id not in sessions[i].results:
                    i -= 1
                sessions[i].click(document_id)
    return sessions

class ClickModel(ABC):

    @abstractmethod
    def click(self, relevances):
        pass

class RCM(ClickModel):

    def __init__(self, max_at=3):
        self.max_at=3
        self.p = 0.5
        self.estimate_parameters(parse_log("./YandexRelPredChallenge.txt"))

    def click(self, relevances):
        return np.random.binomial(1, self.p, size=len(relevances))

    def estimate_parameters(self, sessions, at=10):
        n_shown_docs = 0
        n_clicks = 0
        for s in sessions:
            n_shown_docs += len(s.clicks_rank)
            n_clicks += sum(s.clicks_rank)
        self.p = n_clicks / n_shown_docs

class PBM(ClickModel):
    def __init__(self, at=AT, p_attraction=0.95, file_p_examinations=None):
        self.p_attraction = p_attraction
        self.p_examination = np.random.random(at)
        if file_p_examinations != None:
            with open(file_p_examinations, "r") as f:
                self.p_examination = json.load(f)

    def click_probabilities(self, relevances):
        p_attraction = np.array([self.p_attraction if relevance == 1 else 1-self.p_attraction for relevance in relevances])
        cDist = p_attraction * self.p_examination[:len(relevances)]
        return cDist

    def simulate_first_click(self, relevances):
        cDist = self.click_probabilities(relevances)
        cDist /= cDist.sum()
        chosen_ranking = np.random.choice(len(relevances), p = cDist)
        return chosen_ranking

    #Returns list of boleeans stating if a document was clicked
    def click(self, relevances):
        cDist = self.click_probabilities(relevances)
        return np.random.binomial(1, cDist, size=len(relevances))

    #initial_attraction/initial_examination is a function that returns the initial attraction of a query_id, document_id pair
    def estimate(self, sessions, at=10, verbose=False):
        p_examination = [EMX()  for i in range(at)]
        p_attractiveness = {} #Dictionary that maps from (query_id, document_id) to [[sum, count], old_attractiveness]
        old_examination = np.zeros(at)
        tol = 0.001
        while np.mean([abs(p_examination[i].value() - old_examination[i]) for i in range(at)]) > tol:
            old_examination = [p.value() for p in p_examination]
            for s in sessions:
                query_id = s.query_id
                for rank, document_id in enumerate(s.results):
                    attractiveness = p_attractiveness.get((query_id, document_id), EMX())

                    old_attr = attractiveness.value()
                    old_exam = p_examination[rank].value()

                    attractiveness.count += 1.
                    p_examination[rank].count += 1.

                    if s.clicks_rank[rank]:
                        attractiveness.sum += 1.
                        p_examination[rank].sum += 1.
                    else:
                        attractiveness.sum      += (1.-old_exam) * old_attr / (1. - old_attr * old_exam)
                        p_examination[rank].sum += (1.-old_attr) * old_exam / (1. - old_attr * old_exam)
                    p_attractiveness[(query_id, document_id)] = attractiveness

            if verbose:
              print("EXAMINATION")
              for rank in range(at):
                  print('{:.3e}   {:.3e} {:.3e}'.format(p_examination[rank].sum, p_examination[rank].count, p_examination[rank].value()))

              print("ATTRACTION")
              i=0
              for key, value in p_attractiveness.items():
                  if i<1000 and i%100 == 5:
                      print('{:.3e}   {:.3e} {:.3e}'.format(value.sum, value.count, value.value()))
                  i+=1
              print("\n\n\n")
        return [p.value() for p in p_examination]


# In[ ]:


def relevance_combinations(relevances, at=0, max_rel=MAXREL):
    for rel in range(max_rel + 1):
        relevances[at] = rel
        if at + 1 == len(relevances):
            yield relevances
        else:
            yield from relevance_combinations(relevances, at+1)

def generate_relevance_pairs(at=AT):
    for rel_left in relevance_combinations(np.zeros(at)):
        for rel_right in relevance_combinations(np.zeros(at)):
            yield (np.array(rel_left), np.array(rel_right))

def calculate_err(relevances, at=AT, max_rel=MAXREL):
    thetas = (np.power(2, relevances) - 1) / 2**max_rel
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

def generate_document_overlaps(rel_pair, same_as=([None, None, None], [None, None, None]), at=0):
    if at == len(rel_pair[0]):
        yield same_as
        return
    yield from generate_document_overlaps(rel_pair, same_as=same_as, at=at+1)
    for idx, rel_r in enumerate(rel_pair[1]):
        if rel_pair[0][at] == rel_r and idx not in same_as[0]:
            same_as[0][at] = idx
            same_as[1][idx] = at
            yield from generate_document_overlaps(rel_pair, same_as=same_as, at=at+1)
            same_as[0][at] = None
            same_as[1][idx] = None

def interleave(rel_pair, method, k=2000, at=AT):
    softmax = lambda at: 1/(np.arange(at)+1)**3
    if method == 'teamdraft':
        initial_dist = np.ones((2,at),bool)
        chooser = lambda probDistr, r: ([probDistr[r].argmax()], [r])
    elif method == 'probabilistic':
        initial_dist = np.array([softmax(at), softmax(at)])
        chooser = lambda probDistr, r: ([np.random.choice(at, p=probDistr[r]/probDistr[r].sum()),
                                         np.random.choice(at, p=probDistr[r]/probDistr[r].sum())],
                                        [r, not r])
    for same_as, _ in product(generate_document_overlaps(rel_pair), range(k)):
        probDistr = initial_dist.copy()
        relevances, attribution = np.zeros(at), np.zeros(at)
        y = 0
        while y < at:
            R = random.randint(0,1)
            if probDistr[R].sum() == 0:
                R = not R
            idxs, attr = chooser(probDistr, R)
            for _y in range(min(at - y, len(idxs))):
                idx, r = idxs[_y], attr[_y]
                probDistr[r][idx] = 0
                if same_as[r][idx]:
                    probDistr[not r][same_as[r][idx]] = 0
                relevances[y] = rel_pair[r][idx]
                attribution[y] = r
                y += 1
        yield relevances, attribution

def generate_DERR_bins():
    filled_bins = [[] for _ in range(len(BIN_LABELS))] # list of [(relevance_E, relevance_P), ...]. Each list corresponds to the bin with limits in bin_labels
    for relevances_E, relevances_P in generate_relevance_pairs():
        err_E = calculate_err(relevances_E)
        err_P = calculate_err(relevances_P)
        delta_err = err_E - err_P
        bIdx = find_bin_index(delta_err)
        if bIdx != -1:
            filled_bins[bIdx].append((relevances_E, relevances_P))
    return filled_bins

def simulate_online_experiment(rel_pair, click_model, N=70, interleaving_method='teamdraft'):
    victories = [0,0]
    for relevances, attribution in interleave(rel_pair, method=interleaving_method):
        #clicks are a list of booleans that represent whether that rank was clicked or not
        pair_wins_E = 0
        pair_wins_P = 0
        for _ in range(N):
            clicks = click_model.click(relevances) #
            clicksE = sum(click for i, click in enumerate(clicks) if attribution[i]==0)
            clicksP = sum(click for i, click in enumerate(clicks) if attribution[i]==1)
            if clicksE > clicksP:
                victories[0] += 1
                pair_wins_E += 1
            elif clicksP > clicksE:
                victories[1] += 1
                pair_wins_P += 1
    return victories[0]/(victories[0] + victories[1])

def estimate_sample_size(p1, alpha=0.05, beta=0.90, p0=0.5):
    z_alpha = scipy.stats.norm.ppf(1-alpha)
    z_beta = scipy.stats.norm.ppf(beta)
    nDash = ((z_alpha * math.sqrt(p0 * (1-p0)) + z_beta * math.sqrt(p1 * (1-p1))) / abs(p1 - p0))**2
    n = nDash + (1 / abs(p1 - p0))
    return n

def run_experiment(click_model, interleaving):
  bins = generate_DERR_bins()
  for bIdx, _bin in enumerate(bins):
      if len(_bin) == 0:
          print('Empty bin: {}'.format(BIN_LABELS[bIdx]))
          continue
      Ns = []
      pVictorys = []
      for rel_pair in _bin:
          pVictory = simulate_online_experiment(rel_pair, click_model, interleaving_method='teamdraft')
          n = estimate_sample_size(pVictory)
          Ns.append(n)
          pVictorys.append(pVictory)
      Ns = np.array(Ns)
      pVictorys = np.array(pVictorys)
      print('{}\t\tMedianPWins: {: 3.2f}, Min: {: 3.2f}, Median: {: 7.2f}, Max: {: 7.2f}'.format(
          BIN_LABELS[bIdx],
          np.median(pVictorys),
          Ns.min(),
          np.median(Ns),
          Ns.max()))

def main():
    method = sys.argv[1]
    model = sys.argv[2]

    if model == 'PBM':
        click_model = PBM(file_p_examinations="p_examinations.json")
    elif model == 'RCM':
        click_model = RCM()

    run_experiment(click_model, method)

if __name__ == "__main__":
    main()
