#!/usr/bin/env python
# coding: utf-8

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

class initialize:
    
    def __init__(self, cities, city_names, populationSize):
        self.cities = cities
        self.city_names = city_names
        self.populationSize = populationSize
        #self.cityList = cityList
    


    def generatePath(self):
        path = random.sample(self.cities, len(self.cities))
        return path

    def plot_pop(self, pop_plot):
        xList = []
        yList = []
        for i in range(len(self.cities)):
            xList.append(self.cities[i][0])
            yList.append(self.cities[i][1])

        plt.figure(figsize=(20,10))
        x = [i[0] for i in pop_plot]
        y = [i[1] for i in pop_plot]
        x1=[x[0],x[-1]]
        y1=[y[0],y[-1]]
        plt.plot(x, y, 'b', x1, y1, 'b')
        plt.scatter (x, y) 

        for i, txt in enumerate(self.city_names):
            plt.annotate(txt, (xList[i], yList[i]),horizontalalignment='center', 
                verticalalignment='bottom',
                        )
            print(txt, xList[i], yList[i])
        plt.show()
        return

    def initialPopulation(self):
        population = [self.generatePath() for i in range(0, self.populationSize)]
        return population


class GeneticAlgorithm:
    
    #def __init__(self, cities):
        #self.cities = cities
        
    
    def distance_between_cities(self, cities):
        data = dict()
        for index, value in enumerate(cities):
            x1 = cities[index][0]
            y1 = cities[index][1]
            if index + 1 <= len(cities)-1:
                x2 = cities[index+1][0]
                y2 = cities[index+1][1]
                xdiff = x2 - x1
                ydiff = y2 - y1
                dst = (xdiff*xdiff + ydiff*ydiff)** 0.5
                data['Distance from city '+ str(index+1) +' to city ' + str(index+2)] = dst 
            elif index + 1 > len(cities)-1:
                x2 = cities[0][0]
                y2 = cities[0][1]
                xdiff = x2 - x1
                ydiff = y2 - y1
                dst = (xdiff*xdiff + ydiff*ydiff)** 0.5
                data['Distance from city '+ str(index+1) + ' to city ' + str(index +2 -len(cities))] = dst
        return data

    def total_distance(self, cities):
        total = sum(self.distance_between_cities(cities).values())
        return total
    
    #For the fitness levels, I base it off of total distance. We can decide on this, but I get the total distances and 
    #do 1/totalDistance.
    def path_fitness(self, people):
        total_dis = self.total_distance(people)
        fitness = 1 / float(total_dis)
        return fitness
    
    #Rank them bitches
    def rankPaths(self, population):
        #results = []
        results = {}
        for i in range(len(population)):
            results[i] = self.path_fitness(population[i])
            #results.append(self.path_fitness(population[i]))             
        #This just sorts them in order from greated fitness to least fitness
        # population #: fitness value. There are 10 in total because made 10 populations            
        #combination = [list(a) for a in zip(results, population)]        
        return sorted(results.items(), key = operator.itemgetter(1), reverse = True)
        #return sorted(combination, key = operator.itemgetter(0), reverse = True)
    
    #This pretty much just orders the thing correctly using the selected values
    def get_pool_for_sex(self, population, elites):
        return [population[elites[i]] for i in range(len(elites))]
  
    
    def haveSex(self, father, mother):
        #father and mother are just names I gave them
        #father would be the poolForSex, but randomized in order and mother would be 
        #the same but starting from the end. I try to start from the end and then meet in the middle
        #So Father is first map when i = 0 and mother would be map at i = len(pool) - i -1
        
        #The method of max and min I found online. It's just a way to get father's population and mother's population
        
        #Generate two random lengths for generation 1 and generation 2
        gen1Length= int(random.random() * len(father))
        #print(gen1Length, 'Gen1')
        gen2Length = int(random.random() * len(mother))
        #print(gen2Length, 'GEN2')
        first_generation = min(gen1Length, gen2Length)       
        last_generation = max(gen1Length, gen2Length)
        
        #print(father, "FAATHER")
        #This just get's the indicated "i" value from father's population
        
        #So here is where the bredding stuff goes down. I left it printed. Go to where you see numbers of Gen1 and Gen2
        #Youll see that some are different. Okay so look at father. And i would be starting at whatever the first generation is
        #So if gen1Lenth is 2, the it would start at father[2] and go all the way to say father [5] depending on what that value is
        #Example: range(2,5)
        #So it adds those CITIES, NOT MAPS. remember we are doing this whole function for the length of the "losers". So if it is 5, then each time
        #the function is run is a new map. So right now we are at the first run of the function, for the first loser map.
        
        #So for this map, the indicated "i" cities are added to tot_parent1. For tot_parent2, we add stuff from MOTHER, not FATHER like how we did in tot_parent1
        #So just add whatever remaining space is left. This is sort of like how VIctoria did the mutate.
        #It was inspired by that. NOTE that this is NOT mutating, this is simply breedidng and combining features of both because sex obv.
        tot_parent1 = [father[i] for i in range(first_generation, last_generation)]
        #print(mother, "MOOTHER")
        #print(tot_parent1)
        #Whatever isn't from father, include to here
        tot_parent2 = [i for i in mother if i not in tot_parent1]
        #print(tot_parent2)
        
        #Returns the combination of the two for ONE MAP. For i=2 in range(LosingLength), new values for gen1length and gen2length are formed and the same thing is done.
        #In the end, we should have run this 5 times if the elites was 5. Or 9 times if the elites was 1.
        return tot_parent1 + tot_parent2
        
        
    def population_after_sex(self, poolForSex, elites):
        #This is the length of the population that is not part of the elites
        LosingLength = len(poolForSex) - elites
        #print("-------------POPULATION AFTER SEX FUNCTION----------------")
        #This just randomizes order of the pool and makes Good the pool's length
        GoodLength = random.sample(poolForSex, len(poolForSex))
        #print(GoodLength, "GOOD LENGTH")
        #print("POOL FOR SEX", poolForSex)
        #This is the first part of the population of only the elites. So need part 2 of the people to have sex to survive
        #This hold 5 maps, NOT 5 cities, depending on the elites value. If elites was 2, part1 hold 2 MAPS.
        part1 = [poolForSex[i] for i in range(elites)]
        #print("PART 1!!!!!!!!!!!", part1)
        #The people who have sex are the people in the "New Elite Population". This is because if you are not elite, you don't survive. Duhhh
        #"Battle of the Fittest".
        #This is why when we "perform selection" the stuff in the for loop doesn't really matter. They will procreate.
        #The 5 elite maps stay the same. Compare the first 5 maps from poolForSex and part1 to see
        #The people mating are the 5 maps but for the length of the losers. Here maps mate with each other essentially
        #For example, if elites was 3, then part1 would have 3 maps and part 2 would have 7 maps
        
        #Runs this for the remaining of the spaces. Look at explanation in function. It is run for len(pool) - elites times!!!
        part2 = [self.haveSex(GoodLength[i], GoodLength[len(poolForSex)-i-1]) for i in range(LosingLength)]
        combined = part1 + part2
        #Returns the population of maps after having sex.
        return combined
    
    #Need to fix to work with functions.
    def mutate(self, individual_city, rate):
        #This just chooses 1 random position in the population, so like 2 and exchanges the "exchanged" value:those cities. 
        #So exchanged will be 0,1,2,3,4,...,9 because for loop remember. "exchanged_with" is the rnadom position. What is returned
        #is the population with those cities changed. This happens for 10 times, so the length of the map, or popuation.
        #This function is run 10 times, each for each map. And for a map, it is also run 10 times, each for cities.
        #So if exchanged is 2 and exchanged_with is 7 and those two potisions change. But this goes on for 10 times
        mutation_pick = random.random()
        if  mutation_pick < rate:
            for exchanged in range(len(individual_city)):
                exchanged_with = int(random.random() * len(individual_city))
                #print('\n')
                #print(exchanged, 'exchanged')
                #print(exchanged_with, 'exchanged with')
            
                city1 = individual_city[exchanged]
                #print(city1, 'city1')
                city2 = individual_city[exchanged_with]
                #print(city2, 'city2')
                #print('\n')
                
                individual_city[exchanged] = city2
            
                individual_city[exchanged_with] = city1
                #print(individual_city, 'individual city')
                
        else:
            #Victoria's thing. Switches 2 random elements. ONLY 2 though. It could be that it tries switching with itself.
            #For this, need to include if-else stament I think?? Or just no mutation at all??
            exchanged = int(random.random() * len(individual_city))
            exchanged_with = int(random.random() * len(individual_city))
            #If they are the same value, then no mutation? So all good?
            #print('\n')
            #print(exchanged, 'exchanged')
            #print(exchanged_with, 'exchanged with')
            
            city1 = individual_city[exchanged]
            #print(city1, 'city1')
            city2 = individual_city[exchanged_with]
            #print(city2, 'city2')
            #print('\n')
                
            individual_city[exchanged] = city2
            
            individual_city[exchanged_with] = city1
            #print(individual_city, 'individual city')
        
        return individual_city

    #This returns the pool of maps after mutating each one. Code is about the same as the get_pool_for_sex() one.
    #Just switched the function being called
    def get_pool_after_mutation(self, population, rate):
        return [self.mutate(population[i], rate) for i in range(len(population))]
    
    #This funciton I found somewhere on StackOverFlow. The df parts at least.
    #Basically, we pass in an eliteSize, so like "Top 3" or "Top 5"
    #So the Top 5 populations, or maps, are kept intact. Meaning they survive for sure.
    #The remaining 5 are selected randomly, could be repeated. So we can have repeated maps. This doesn't matter because
    #We don't really save them anywhere (yet). And since we mutate and stuff it'll be different anyway.
    #And it helps because then we wont "run" out of maps.
    #I'm not sure if we NEED it, but I am using it for now to see where I go from here. Maybe it helps in recognizing the "best" map
    def perform_selection(self, pop, eliteSize): #Again, I found this on stack overflow and modified it
        #output = rankPathes(population)
        df = pd.DataFrame(np.array(pop), columns=["Index","Fitness"])
        #print(df)
        #A cumulative sum is a sequence of partial sums of a given sequence
        df['cumulative_sum'] = df.Fitness.cumsum()
        #print(df['cumulative_sum'], "cum sum!!!!")
        #Cumulative percentage is another way of expressing frequency distribution. 
        #It calculates the percentage of the cumulative frequency within each interval, much as relative frequency distribution calculates the percentage of frequency.
        df['cum_percentage'] = 100*df.cumulative_sum/df.Fitness.sum()
        selected_values = [pop[i][0] for i in range(eliteSize)] #The 5 "maps" of the top 5 ranks. On ranks, the first value of the tuple
            
        #print(selected_values, '!st selection!!')
        #Now pick the remaining randomly. Can be repeated "maps" because they are not considered elit
        for i in range(len(pop) - eliteSize):
            pick = 100*random.random()
            #print("pick1!!", pick)
            for i in range(0, len(pop)):
                if pick <= df.iat[i,3]:
                    #print(df.iat[i,3])
                    selected_values.append(pop[i][0])
                    break
            #Returns map #'s or map ID's. So like Map2, Map5, Map1, etc. ID's!!! NOT ENTIRE MAPS!!!!    
        return selected_values

    def next_generation(self, population, elites, rate):
        #Population would start with the inital one and then each time after the mutation, it will change
        ranks = self.rankPaths(population)
        selections = self.perform_selection(ranks, elites)
        poolForSex = self.get_pool_for_sex(population, selections)
        afterSex = self.population_after_sex(poolForSex, elites)
        mutated_pool = self.get_pool_after_mutation(afterSex, rate)
        #print(mutated_pool, "MUTATED")
        return mutated_pool
    
    #Generations meaning how many generations to run the program for
    def pass_time(self, elites, rate, generations):
        f = open("TSM.txt", 'r').read().splitlines()
        cityCoords = np.array([ tuple( map( float, coord.split() ) ) for coord in f ]).tolist()
        city_names = open("city_names.txt", 'r').read().splitlines()
        popSize = 10
        initial = initialize(cityCoords, city_names, popSize)
        population = initial.initialPopulation()
        #Runs the thing generations times
        #After each run, the optimal route and stuff recognizes
        for i in range(generations):
            population = self.next_generation(population, elites, rate)
            print("Population",i,population)
            print('\n')
        best_rank = self.rankPaths(population)[0][0] #Has to be outside cause population[0][0] doesnt work.
        optimal_route = population[best_rank]
        #ordered_cities = self.get_names(optimal_route,cityCoords,city_names)
        #print([(indx,val) for indx,val in enumerate(ordered_cities)])
        #plot_pop(optimal_route)
        print(optimal_route)
        initial.plot_pop(optimal_route)
        return optimal_route


if __name__ == '__main__':

    f = open("TSM.txt", 'r').read().splitlines()
    cityCoords = np.array([ tuple( map( float, coord.split() ) ) for coord in f ]).tolist()
    city_names = open("city_names.txt", 'r').read().splitlines()
    popSize = 10
    initial = initialize(cityCoords, city_names, popSize)
    population = initial.initialPopulation()
    
    for idx, pop_plot in enumerate(population):
        print('Init pop. ' + str(idx), pop_plot)
        print('\n')
        initial.plot_pop(pop_plot)

    genetics = GeneticAlgorithm()
    fitness = genetics.path_fitness(cityCoords)
    ranks = genetics.rankPaths(population)
    


    genetics = GeneticAlgorithm()

    #selected_values = genetics.perform_selection(ranks, 5)
    
    #This is the number of "fit" people per generation. Can change it
    elites = 5
    rate = random.uniform(0,1)

    #popp = genetics.rankPaths(population)
    
    #This returns the top 5 values AND selections of any of the people in the rankings.
    #So returns the values likes [0,2, 3,4,5,6,4,5,3,1]
    #Where the first 5 in the list are the top 5 rankings of the population (look at ranks)
    #Andd the rest are selections of ANY remaining 5, can be repeated but it doesn't matter because we will mate
    selected_values = genetics.perform_selection(ranks,elites)
    selected_values.pop(0)
    selected_values.pop(1)
    selected_values.pop(2)
    #This returns the population pool that will be used to mate using the selected values
    poolForSex = genetics.get_pool_for_sex(population, selected_values)
    
    #The population after having sex
    after = genetics.population_after_sex(poolForSex, elites)
    #print("-----------------------------BEFORE SEX-----------------------")
    #print(poolForSex)

    #print("-----------------------------AFTER SEX-----------------------")
    #print(after)
    #print("-----------------------------AFTER MUTATION-----------------------")
    
    '''Use the next two lines to see how it works with ONE map!!!!
    genetics.mutate(after[0], rate)
    print(rate, "rate")'''
    
    mutated_pool = genetics.get_pool_after_mutation(after, rate)
    
    '''test_mutate_function = genetics.next_generation(population, elites, rate)''' #Works
    
    #best_route = genetics.pass_time(elites, rate, 10000)

    #Now to call the whole thing again but for X times, or X generations