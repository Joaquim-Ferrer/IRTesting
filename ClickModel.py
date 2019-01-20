import random
import numpy as np
from YandexParser import parseYandexLog
import Session

#CHECK IF MAX AT IS OK WITH SIMULATE CLICK AND STUFF AS WE ARE TRAINING FOR 10 RANKINGS AND USING ONLY MAX 6
class EMX():
    PROB_MIN = 0.000001
    def __init__(self, sum=1., count=2.):
        self.sum = sum
        self.count = count

    def value(self):
        return min(self.sum / float(self.count), 1 - self.PROB_MIN)

class BinaryRelevancePBM:
    def __init__(self, max_at=6, p_attraction=0.9):
        self.max_at = max_at
        self.p_examination = np.random.random(max_at)
        self.p_attraction = p_attraction

    def click_probabilities(self, relevances):
        p_attraction = np.array([self.p_attraction if relevance == 1 else 1-self.p_attraction for relevance in relevances])
        cDist = p_attraction * self.p_examination[:len(relevances)]
        return cDist

    def simulate_click(self, relevances):
        cDist = self.click_probabilities(relevances)
        cDist /= cDist.sum()
        chosen_ranking = np.random.choice(len(relevances), p = cDist)
        return chosen_ranking

    #initial_attraction/initial_examination is a function that returns the initial attraction of a query_id, document_id pair
    def estimate_parameters(self, sessions, n_iterations=100, at=10):
        p_examination = [EMX()  for i in range(at)]
        p_attractiveness = {} #Dictionary that maps from (query_id, document_id) to [[sum, count], old_attractiveness]

        for i in range(n_iterations):
            for s in sessions:
                query_id = s.query_id
                for rank, document_id in enumerate(s.results):
                    attractiveness = p_attractiveness.get((query_id, document_id), EMX())

                    oattr = attractiveness.value()
                    oexam = p_examination[rank].value()

                    attractiveness.count += 1.
                    p_examination[rank].count += 1.

                    if s.clicks_rank[rank]:
                        attractiveness.sum += 1.
                        p_examination[rank].sum += 1.
                    else:
                        attractiveness.sum      += (1.-oexam) * oattr / (1. - oattr * oexam)
                        p_examination[rank].sum += (1.-oattr) * oexam / (1. - oattr * oexam)
                    p_attractiveness[(query_id, document_id)] = attractiveness

            print("EXAMINATION")
            #Update old values and reset count and sum
            for rank in range(at):
                print('{:.3e}   {:.3e} {:.3e}'.format(p_examination[rank].sum, p_examination[rank].count, p_examination[rank].value()))


            print("ATTRACTION")
            i=0
            for key, value in p_attractiveness.items():
                if i<1000 and i%100 == 5:
                    print('{:.3e}   {:.3e} {:.3e}'.format(value.sum, value.count, value.value()))
                i+=1
            print("\n\n\n")
