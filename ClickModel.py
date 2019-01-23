import random
import numpy as np
from YandexParser import parseYandexLog
import Session
import json
from abc import ABC, abstractmethod

#CHECK IF MAX AT IS OK WITH SIMULATE CLICK AND STUFF AS WE ARE TRAINING FOR 10 RANKINGS AND USING ONLY MAX 6
class EMX():
    def __init__(self, sum=1., count=2.):
        self.sum = sum
        self.count = count

    def value(self):
        return min(self.sum / float(self.count), 0.999999)

class ClickModel(ABC):

    @abstractmethod
    def get_clicks(self, relevances):
        pass

class RCM(ClickModel):
    
    def __init__(self, max_at=3):
        self.max_at=3
        self.p = 0.5
        self.estimate_parameters(parseYandexLog("./YandexRelPredChallenge.txt"))
    
    def get_clicks(self, relevances):
        return np.random.binomial(1, self.p, size=len(relevances))

    def estimate_parameters(self, sessions, at=10):
        n_shown_docs = 0
        n_clicks = 0
        for s in sessions:
            n_shown_docs += len(s.clicks_rank)
            n_clicks += sum(s.clicks_rank)
        self.p = n_clicks / n_shown_docs

class BinaryRelevancePBM(ClickModel):
    def __init__(self, max_at=3, p_attraction=0.95, file_p_examinations=None):
        self.max_at = max_at
        self.p_attraction = p_attraction
        self.p_examination = np.random.random(max_at)
        if file_p_examinations != None:
            with open(file_p_examinations, "r") as f:
                self.p_examination = json.load(f)

    def get_click_probabilities(self, relevances):
        p_attraction = np.array([self.p_attraction if relevance == 1 else 1-self.p_attraction for relevance in relevances])
        cDist = p_attraction * self.p_examination[:len(relevances)]
        return cDist

    def simulate_first_click(self, relevances):
        cDist = self.get_click_probabilities(relevances)
        cDist /= cDist.sum()
        chosen_ranking = np.random.choice(len(relevances), p = cDist)
        return chosen_ranking

    #Returns list of 0/1 stating if a document was clicked
    def get_clicks(self, relevances):
        cDist = self.get_click_probabilities(relevances)
        return np.random.binomial(1, cDist, size=len(relevances))

    #initial_attraction/initial_examination is a function that returns the initial attraction of a query_id, document_id pair
    def estimate_parameters(self, sessions, at=10):
        p_examination = [EMX()  for i in range(at)]
        p_attractiveness = {} #Dictionary that maps from (query_id, document_id) to [[sum, count], old_attractiveness]
        old_examination = np.zeros(at)
        tol = 0.00001
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

def main():
    sessions = parseYandexLog("./YandexRelPredChallenge.txt")
    print('PARSED YANDEX CLICK LOG')
    Pbm = BinaryRelevancePBM(3)
    prob_examination = Pbm.estimate_parameters(sessions)
    with open('p_examinations.json', 'w') as fp:
        json.dump(prob_examination, fp)

if __name__ == '__main__':
    main()
