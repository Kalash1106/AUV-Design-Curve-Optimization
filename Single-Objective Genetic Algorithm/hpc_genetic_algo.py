import pygad #Code library for genetic algorithm. This file requires pyGAD to be installed https://pygad.readthedocs.io/en/latest/ 
from argparse import ArgumentParser
import numpy as np
import hpc_design_curve as auv
import drag
import numba
import time

#Constructing the initial population space
# We have a total of ~ 3 million possible combinations
#The beginning sample population space is 4375
def make_population(g_list : list):
    for r_max in np.arange(0.25, 0.5, 0.1):
        for nn in np.arange(2, 5, 0.5):
            for nt in np.arange(2, 5, 0.5):
                for ln in np.arange(0.25, 0.7, 0.1):
                    for lt in np.arange(0.25, 0.7, 0.1):
                        g_list.append([r_max, nn, nt, ln, lt])

    return g_list

#Constructing the gene space dictionary
def make_genes(genes_list : list):
    genes_list.append(list(np.arange(0.25, 0.5, 0.01)))
    genes_list.append(list(np.arange(2, 5, 0.2)))
    genes_list.append(list(np.arange(2, 5, 0.2)))
    genes_list.append(list(np.arange(0.25, 0.7, 0.01)))
    genes_list.append(list(np.arange(0.25, 0.7, 0.01)))

    return genes_list

#The fitness function derived from ansys simulations
def fitness_func(solution : list, solution_idx):
    r_max = solution[0]
    nn = solution[1]
    nt = solution[2]
    ln = solution[3]
    lt = solution[4]
    return 1/drag.total_drag(r_max, nn, nt, ln, lt)
    

if __name__ == '__main__':
    parser = ArgumentParser(description="The Argument Parser for AUV Optimisation")

    parser.add_argument('--ng', type= float, default= 50, help='Number of generations')
    parser.add_argument('--nm', type= float, default= 4, help='Number of parents mating')
    parser.add_argument('--pst', type = str , default= "sss", help='Parent Selection type')
    parser.add_argument('--cot', type = str , default= "single_point", help='Cross over type')

    arg_d = vars(parser.parse_args()) #The argument dictionary

    #Gene space 
    g_list = []
    g_list = make_population(g_list)
    genes_list = []
    genes_list = make_genes(genes_list)
    parent_selection_type = arg_d['pst']
    crossover_type = arg_d['cot']
     
    fitness_function = fitness_func

    s1 = time.perf_counter()
    ga_instance = pygad.GA(
                    num_generations = arg_d['ng'],
                    parallel_processing=["thread", 5],
                    num_parents_mating = arg_d['nm'],
                    fitness_func=fitness_function,
                    initial_population = g_list, 
                    gene_space=genes_list,
                    parent_selection_type = parent_selection_type,
                    crossover_type = crossover_type,
                    mutation_type = "random",
                    mutation_num_genes=1
                    )
    
    #Running the ga_instance and solving the problem
    ga_instance.run()
    s2 = time.perf_counter()

    #The best solution and the parameters
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(" ")
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {} N".format(1/solution_fitness))
    print("Time taken to run the program is", s2 - s1, 's')

     



          


    






