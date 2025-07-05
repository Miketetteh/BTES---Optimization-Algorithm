import numpy
import os
import matplotlib.pyplot as plt
from pandas import DataFrame

import ifm
from ifm import Enum
import numpy as np
import math



def cal_pop_fitness(pop):
    fitness = []
    for n in range(pop.shape[0]):
        # Calculating the fitness value of each solution in the current population.
        ifm.getKernelVersion()
        ifm.forceLicense(
            'user= michael.tetteh@mines.sdsmt.edu, passwd=MichT@MIKE23')  # replace with your own info , options=FH3

        os.chdir('C:/Users/Michael Tetteh/Downloads/btes_fifth')
        os.getcwd()

        d_winter1 = 183    # Number of days for extracting energy
        d_summer1 = 182    # Number of days for injecting energy

        femwin1_1 = 'my_project_mike (3).fem'
        femsum1_1 = 'my_project_mike (3).fem'

        dacwin1_1 = 'temp_result/my_project_mike'
        dacsum1_1 = 'temp_result/my_project_mike'

        yr_cycle = 3        # Number of year cycle
        n_nodes = 6503 # update with your own number of nodes
        #n_element = 10554


        BHE_in = []
        BHE_out = []
        time_list = []

        dc = ifm.loadDocument(femwin1_1)
        # for k in range(n_element):
            # oc.setMatHeatSolidConductivity(k,cond[0])

        for iyr in range(yr_cycle):
            # Start the charging simulation
            print('summer1_1_y' + str(iyr + 1))
            doc = ifm.loadDocument(femsum1_1)
            #for k in range(n_element):
                #doc.setMatHeatSolidConductivity(k, pop[n][0] * 86400)
            #for m in range(37):
                #doc.setBHEFlowDischarge(m, pop[n][1])

            doc.setBHEArrayFlowDischarge(0, pop[n][0])    # Set the charging flowrate parameters to the feflow model

            # In order to use the same model for the entire optimization process, the temperature was always reset to
            # the initial temperature which is "10" in this work before starting a new simulation cycle
            if iyr == 0:
                t_fin = 0
                temperature = []
                for i in range(n_nodes):
                    temperature.append(10)

            for i in range(n_nodes):
                doc.setResultsTransportHeatValue(i, temperature[i])
            # set "final" and  "initial" simulation time
            doc.setInitialSimulationTime(t_fin)
            doc.setFinalSimulationTime(t_fin + d_summer1)

            # for k in range(n_element):
                    # doc.setMatHeatSolidConductivity(k,cond[j])
            # doc.saveDocument()
            doc.saveDocument()
            dacname = dacsum1_1 + str(iyr + 1) + '.dac'
            doc.setOutput(dacname, [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 1095])  # Save the output result in a "dac" file
            doc.startSimulator()
            doc.stopSimulator()
            doc.closeDocument()

            # Extract the output result to be used to compute the fitness (ie, the inlet and outlet temperature, the time)
            doc = ifm.loadDocument(dacname)
            t_fin = doc.getFinalSimulationTime()
            temperature = []
            for i in range(n_nodes):
                temperature.append(doc.getResultsTransportHeatValue(i))
            BHE_Data = doc.getHistoryValues(Enum.HIST_BHE_IO)
            for i in (BHE_Data[1][26]):
                BHE_in.append(i)
            for i in (BHE_Data[1][27]):
                BHE_out.append(i)

            for i in (BHE_Data[0]):
                time_list.append(i)

            doc.closeDocument()

            # Start the injection simulation

            print('winter1_1_y' + str(iyr + 1))
            doc = ifm.loadDocument(femsum1_1)

            # set initial temperature from previous period
            for i in range(n_nodes):
                doc.setResultsTransportHeatValue(i, temperature[i])
            # set "final" and  "initial" simulation time
            doc.setInitialSimulationTime(t_fin)
            doc.setFinalSimulationTime(t_fin + d_winter1)

            # Set the discharging flowrate parameters in the feflow model
            doc.setBHEArrayFlowDischarge(0, pop[n][1])
                # doc.saveDocument()
            doc.saveDocument()
            dacname = dacsum1_1 + str(iyr + 1) + '.dac'
            doc.setOutput(dacname, [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 1095])
            doc.startSimulator()
            doc.stopSimulator()
            doc.closeDocument()

            # Extract the ouput data

            doc = ifm.loadDocument(dacname)
            BHE_Data = doc.getHistoryValues(Enum.HIST_BHE_IO)
            for i in (BHE_Data[1][26]):
                BHE_in.append(i)
            for i in (BHE_Data[1][27]):
                BHE_out.append(i)

            for i in (BHE_Data[0]):
                time_list.append(i)

            t_fin = doc.getFinalSimulationTime()
            temperature = []
            for i in range(n_nodes):
                temperature.append(doc.getResultsTransportHeatValue(i))
            doc.closeDocument()

        # Compute the fitness function (Recovery efficiency)

        heat_cap = 4.2E6  # J/m3/K (Volumetric heat capacity)
        fluxC2 = pop[n][1]   # m3/d (Discharging flowrate)
        fluxC3 = pop[n][0]  # m3/d  (Charging flowrate)

        tempa = []
        tempb = []
        tempc = []
        tempd = []

        # Calculate the amount of heat stored and retrieved at each year cycle
        for i in range(len(time_list) - 1):

            temp1 = fluxC3 * heat_cap * BHE_in[i]  # Inlet heat storage
            temp2 = fluxC2 * heat_cap * BHE_in[i]   # Inlet heat lost
            temp1_ = fluxC3 * heat_cap * BHE_out[i]   # Outlet heat retrieved
            temp2_ = fluxC2 * heat_cap * BHE_out[i]     # Outlet heat lost
            tempa.append(temp1)
            tempb.append(temp2)
            tempc.append(temp1_)
            tempd.append(temp2_)

        # Remove any null data from the list
        tempa = [x for x in tempa if not math.isnan(x)]
        tempb = [x for x in tempb if not math.isnan(x)]
        tempc = [x for x in tempc if not math.isnan(x)]
        tempd = [x for x in tempd if not math.isnan(x)]

        half1 = sum(tempa[:1913])/ 1.0E9     # First year oulet heat storage
        half1_ = sum(tempc[:1913])/ 1.0E9    # First year inlet heat lost
        half2 = sum(tempb[1913:])/ 1.0E9     # First year heat retrieved
        half2_ = sum(tempd[1913:])/ 1.0E9    # First year oulet heat lost
        half3 = sum(tempa[518:755])/ 1.0E9   # Second year heat stored
        half3_ = sum(tempc[518:753])/ 1.0E9   # Second year inlet heat lost
        half4 = sum(tempb[753:991])/ 1.0E9    # Second year heat retrieved
        half4_ = sum(tempd[753:991])/ 1.0E9   # Second year oulet heat lost
        half5 = sum(tempa[991:1228])/ 1.0E9   # Third year oulet heat lost
        half5_ = sum(tempc[991:1228])/ 1.0E9    # Third year inlet heat lost
        half6 = sum(tempb[1228:])/ 1.0E9        # Third year heat retrieved
        half6_ = sum(tempd[1228:])/ 1.0E9       # Third year oulet heat lost

        diff1 = half1 - half1_
        diff2 = half2_ - half2
        diff3 = half3 - half3_
        diff4 = half4_ - half4
        diff5 = half5 - half5_
        diff6 = half6_ - half6

        RR1 = diff2 / diff1   # Recovery efficiency for first year of simulation
        RR2 = diff4 / diff3   # Recovery efficiency for second year of simulation
        RR3 = diff6 / diff5   # Recovery efficiency for third year of simulation
        fitness.append(RR3)

    return fitness

# Genetic algorithm

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover