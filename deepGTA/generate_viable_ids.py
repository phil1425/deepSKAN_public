import numpy as np
from .utils import generate_binary_matrix
import itertools

# This module creates all 1024 different kinetic models and 
# excludes them if
# - they are isomorphic to an existing model
# - they have a branching with more than 2 species
# - one of the states does not get populated, after excitation,
# but still has a decay constant

# this is done once after importing this module

num_s = 5

n_permutable = [0, 1, 3, 6, 10][num_s-1]
permutations = 2**n_permutable
num_unique_matrices = 0

unique_ids = []
rest_ids = []

for k_id in range(permutations):
    K = generate_binary_matrix(k_id, num_s)
    K_ = K.copy()
    for i in range(num_s):
        K_[i][i] = 0
    
    encountered_zero = False
    is_unique_matrix = True
    sum_lines = [np.sum(x) for x in K_]
    sum_rows = [np.sum(x) for x in K_.transpose()]

    for i in range(1,num_s):
        if sum_lines[i] == 0:
            if np.sum(sum_lines[i:-1]) > 0:
                is_unique_matrix = False

            if sum_rows[i] > 0:
                is_unique_matrix = False

    for x in sum_rows:
        if x > 2:
            is_unique_matrix = False
    
    if is_unique_matrix:
        num_unique_matrices += 1
        unique_ids.append(k_id)
    else:
        rest_ids.append(k_id)


unique_graphs = []
name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G'][:num_s]
permutable_list = name_list[1:-1]
permutations_list = list(itertools.permutations(permutable_list))
permuted_name_list = [['A']+list(p)+[name_list[-1]] for p in permutations_list]

unique_ids_2 = []

for k_id in unique_ids:
    is_unique = True
    K = generate_binary_matrix(k_id, num_s)
    K_ = K.copy()

    for i in range(num_s):
        K_[i][i] = 0

    graphs_for_one_model = []
    for p in permuted_name_list:
        graph_list = [[], [], [], [], [], []][:num_s]
        for i in range(num_s):
            for j in range(num_s):
                if K_[i][j] > 0:
                    graph_list[i].append(p[j])
        graphs_for_one_model.append(graph_list)

    for graph in graphs_for_one_model:
        if graph in unique_graphs:
            is_unique = False
    
    if is_unique:
        for g in graphs_for_one_model:
            unique_graphs.append(g)
        unique_ids_2.append(k_id)

unique_ids = unique_ids_2
