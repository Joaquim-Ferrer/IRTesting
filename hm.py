import numpy as np
import random

at = 3
max_relevance = 1
#Bin labels include the first and exclude the last.
bin_labels = [[0.05, 0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.8,0.9], [0.9,0.95]]

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
        relevances_A = [0 for i in range(at)]
        if not next_combination(relevances_B):
            return False
    return True

def get_prob_from_relevance(relevance):
    return (2**relevance - 1) / 2**max_relevance

def calculate_ERR(relevances):
    thetas = [get_prob_from_relevance(r) for r in relevances]
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

#Recursively generates all possible interleavings according to the documents on the list being the same or not.
#The result is a list(every possible interleaving) of lists of (relevance, 0/1) where 0 corresponds to a document of A and 1 corresponds to a document of B
def generate_team_draft_interleavings(relevances_A, relevances_B):
    len_A = len(relevances_A)
    len_B = len(relevances_B)
    res = []

    if len_A == 0 and len_B == 0:
        return []
    
    relevances = [relevances_A, relevances_B]
    if len_A > len_B or (len_A == len_B and random.random() > 0.5):
        chosen_list = 0
    else:
        chosen_list = 1

    non_chosen_list = (chosen_list + 1) % 2
    top_result = relevances[chosen_list].pop(0)
    res_beginning = (top_result, chosen_list)
    res_end = []
    #If no document in B matches to that document in A then we don't change the lists
    res_end.extend(generate_team_draft_interleavings(relevances_A, relevances_B))
    #Now we take off the elements from the other list that may be the same as the chosen document and generate all the interleavings.
    for i in range(len(relevances[non_chosen_list])):
        if relevances[non_chosen_list][i] == top_result:
            removed_relevance = relevances[non_chosen_list].pop(i)
            res_end.extend(generate_team_draft_interleavings(relevances_A, relevances_B))
            relevances[non_chosen_list].insert(i, removed_relevance)
    
    #print(res_beginning, res_end)
    #Now we reconstruct all the possible endings of interleaving with the result got in this function call
    if len(res_end) == 0:
        res = [[res_beginning]] 
    for interleaving in res_end:
        temp = [res_beginning]
        temp.extend(interleaving) 
        res.append(temp)    

    relevances[chosen_list].insert(0, top_result)
    return res


#Generate pairs per bin
bins = [[] for i in range(len(bin_labels))] # list of [(relevance_E, relevance_P), ...]. Each list corresponds to the bin with limits in bin_labels
relevances_E = [0 for i in range(at)]
relevances_P = [0 for i in range(at)]
while True:
    err_E = calculate_ERR(relevances_E)
    err_P = calculate_ERR(relevances_P)
    #print(relevances_E, err_E, relevances_P, err_P)
    delta_err = err_E - err_P
    bin_i = find_bin_index(delta_err)
    #print(delta_err, bin_labels[bin_i] if bin_i != -1 else "nada")
    if bin_i != -1:
        fat_object = (relevances_E[:], relevances_P[:]) #Shallow copies as we modify the list when generating the next element
        bins[bin_i].append(fat_object)
    #Generate next pair
    if not next_pair_combination(relevances_E, relevances_P):
        break

for b_i, bini in enumerate(bins):
    print(bin_labels[b_i])
    for pair in bini:
        print(pair)
        interleavings = generate_team_draft_interleavings(pair[0], pair[1])
        for interleaving in interleavings:
            print(interleaving)
            pass
        print("\n\n\n")
            