import random
import numpy as np
from YandexParser import parseYandexLog
import Session

#CHECK IF MAX AT IS OK WITH SIMULATE CLICK AND STUFF AS WE ARE TRAINING FOR 10 RANKINGS AND USING ONLY MAX 6

class BinaryRelevancePBM:

    def __init__(self, max_at=6, p_attraction=0.9):
        self.max_at = max_at
        self.p_examination = np.asarray([random.random() for i in range(max_at)])
        self.p_attraction = p_attraction

    def generate_click_probabilities(self, relevances):
        p_attraction = np.asarray([self.p_attraction if relevance == 1 else 1-self.p_attraction for relevance in relevances])
        click_probabilities = np.multiply(p_attraction, self.p_examination[:len(relevances)])
        return click_probabilities

    def simulate_click(self, relevances):
        click_probabilities = self.generate_click_probabilities(relevances)
        #Normalize click probabilities
        click_probabilities = click_probabilities / sum(click_probabilities)
        chosen_ranking = np.random.choice([i for i in range(relevances)], p = click_probabilities)
        return chosen_ranking

    #initial_attraction/initial_examination is a function that returns the initial attraction of a query_id, document_id pair
    def estimate_parameters(self, sessions, n_iterations=100, initial_attraction=lambda: 1, initial_examination=lambda: 0.5):
        p_examination = [[[0, 0], initial_examination()]  for i in range(10)] #List of [[sum, count], old_examination] for every rank
        p_attractiveness = {} #Dictionary that maps from (query_id, document_id) to [[sum, count], old_attractiveness]
        
        for i in range(n_iterations):
            #print([p_examination[i][1] for i in range(len(p_examination))])
            for s in sessions:
                query_id = s.query_id
                for rank, document_id in enumerate(s.results):
                    #Initialize attractiveness
                    if (query_id, document_id) not in p_attractiveness:
                        p_attractiveness[(query_id, document_id)] = [[0, 0], initial_attraction()]
                    #Get values so that it doesn't become a dictionary indexing mess
                    old_attractiveness = p_attractiveness[(query_id, document_id)][1]
                    old_examination = p_examination[rank][1]

                    p_attractiveness[(query_id, document_id)][0][1] += 1
                    p_examination[rank][0][1] += 1

                    if s.clicks_rank[rank]:
                        p_attractiveness[(query_id, document_id)][0][0] += 1
                        p_examination[rank][0][0] += 1
                    else:
                        p_attractiveness[(query_id, document_id)][0][0] += (1-old_examination   ) * old_attractiveness / (1-old_attractiveness*old_examination)
                        p_examination[rank][0][0] +=                       (1-old_attractiveness) * old_examination    / (1-old_attractiveness*old_examination)
            
            print("EXAMINATION")
            for p in p_examination:
                print(p[0][0], p[0][1], p[1])
            #Update old values and reset count and sum
            for i in range(len(p_examination)):
                p_examination[i][1] = p_examination[i][0][0] / p_examination[i][0][1]
                p_examination[i][0][0] = 0
                p_examination[i][0][1] = 0

            print("ATTRACTION", len(p_attractiveness))
            i=0
            for key, value in p_attractiveness.items():
                if i<1000 and i%100 == 5: 
                    print(value[0][0], value[0][1], value[1])
                i+=1
                #p_attractiveness[key][1] = value[0][0] / value[0][1]
                p_attractiveness[key][0][0] = 0
                p_attractiveness[key][0][1] = 0
            print("\n\n\n")