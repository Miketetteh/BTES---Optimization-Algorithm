import numpy
import ga1



# Inputs of the equation.
#equation_inputs = [4, -2, 3.5, 5, -11, -4.7]

# Number of the weights we are looking to optimize.
num_weights = 2

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 3   # Number of population (Can be changed to a desired population)
num_parents_mating = 2   # Number of parents to select after fitness in computed (Can be changed)

# Defining the population size.
pop_size = (sol_per_pop,
            num_weights)  # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
# Creating the initial population.
new_population = numpy.random.uniform(low=24.0, high=35.0, size=pop_size)
#new_population[:, 0] = numpy.random.uniform(low=1.0, high=5.0, size=sol_per_pop)
print(new_population)

"""
new_population[0, :] = [2.4,  0.7, 8, 2,   5,   1.1]
new_population[1, :] = [0.4, 2.7, 5, 1,   7,   0.1]
new_population[2, :] = [1,   2,   2, 3,   2,   0.9]
new_population[3, :] = [4,    7,   12, 6.1, 1.4, 4]
new_population[4, :] = [3.1,  4,   0,  2.4, 4.8,  0]
new_population[5, :] = [2,   3,   7, 6,   3,    3]
"""

best_outputs = []
num_generations = 30
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = ga1.cal_pop_fitness(new_population)
    print("Fitness")
    print(fitness)


    # Selecting the best parents in the population for mating.
    parents = ga1.select_mating_pool(new_population, fitness,
                                    num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = ga1.crossover(parents,
                                       offspring_size=(pop_size[0], num_weights))    #- parents.shape[0]
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = ga1.mutation(offspring_crossover, num_mutations=2)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:sol_per_pop, :] = offspring_mutation
    #new_population[parents.shape[0]:, :] = offspring_mutation

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = ga1.cal_pop_fitness(new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
#print("Best solution fitness : ", fitness[best_match_idx])