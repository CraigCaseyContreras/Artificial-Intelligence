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
    
    def get_pool_for_sex(self, population, elites):
        return [population[selected_values[i]] for i in range(len(selected_values))]
        
    
    def haveSex(self, father, mother):
        #The method of max and min I found online. It's just a way to get father's population and mother's population
        gen1Length= int(random.random() * len(father))
        print(gen1Length, 'Gen1')
        gen2Length = int(random.random() * len(mother))
        print(gen2Length, 'GEN2')
        first_generation = min(gen1Length, gen2Length)       
        last_generation = max(gen1Length, gen2Length)
        
        print(father, "FAATHER")
        #This just get's the indicated "i" value from father's population
        tot_parent1 = [father[i] for i in range(first_generation, last_generation)]
        print(mother, "MOOTHER")
        print(tot_parent1)
        #Whatever isn't from father, include to here
        tot_parent2 = [i for i in mother if i not in tot_parent1]
        print(tot_parent2)
        return tot_parent1 + tot_parent2
        
        ''''#Randomizes the "order" of the pool. Doesn't matter because we saved the top 5 here so as long as they are here, we good
        pl = random.sample(poolForSex, len(poolForSex))
        ln = len(pl) - elites

        # first and sec are best and 2nd best paths salesman can take
        # first and sec are nested lists containing 10 elements.
        # Each element contains the 1/distance calculation and then the complete path
        first = combo1[0]
        sec = combo1[1]
        # removes the best and 2nd best path from overall list. 
        combo1.pop(0)
        combo1.pop(1)
        # splits nested listing, so now we only have the ordered paths of best and 2nd best.  
        f = first[1]
        s = sec[1]
       
        
        rate = random.uniform(0,1)
        
        mutated_population = self.mutate(f, s, rate)
      
        kiddo = mutated_population[0]
        kiddo2 = mutated_population[1]
        
        # new_population contains ONLY paths (no 1/distance included)
        # it includes unmated children, and the new child. 
        # I programmed it this way so that mating() returns a new population.
        # This new population can then be plugged into path_fitness() and rankPaths()
        new_population = [n[1:] for n in combo1]
        new_population.append(kiddo)
        new_population.append(kiddo2)
        
        return new_population'''
        
    def population_after_sex(self, poolForSex, elites):
        #This is the length of the population that is not part of the elites
        LosingLength = len(poolForSex) - elites
        #Using the elite population, just randomize their order and make that a new population for now of length 5
        GoodLength = random.sample(poolForSex, len(poolForSex))
        #This is the first part of the population of only the elites. So need part 2 of the people to have sex to survive
        part1 = [poolForSex[i] for i in range(elites)]
        #The people who have sex are the people in the "New Elite Population". This is because if you are not elite, you don't survive. Duhhh
        #"Battle of the Fittest".
        #This is why when we "perform selection" the stuff in the for loop doesn't really matter. They won't procreate.
        part2 = [self.haveSex(GoodLength[i], GoodLength[len(poolForSex)-i-1]) for i in range(LosingLength)]
        combined = part1 + part2
        return combined
    
    def mutate(self, f, s, rate):
        kiddo = []
        kiddo2 = []
        
        if rate <= 0.9:
            # Single Point Crossover
            # Pull the first five cities from parent 1, append to list kiddo
            # Pull the remaining cities from parent 2, append to list kiddo
            for l in range(5):
                kiddo.append(f[l])
                f.pop(l)
            for m in range(len(s)):
                if s[m] not in kiddo:
                    kiddo.append(s[m])
            # To keep population from decreasing by 1 every time there is a mating,
            # Mating generates 2 kids. 
            # Pull the first five cities from parent 2, append to list kiddo2
            # Pull the remaining cities from parent 1, append to list kiddo2
            for n in range(5):
                kiddo2.append(s[n])
                s.pop(n)
            for o in range(len(f)):
                if f[o] not in kiddo2:
                    kiddo2.append(f[o])
        elif rate > 0.9:
            # MUTATION
            # Switches 2 elements in best and 2nd best paths
            f[5],f[6] = f[6],f[5]
            kiddo = f
            s[5],s[6] = s[6],s[5]
            kiddo2 = s
        return (kiddo, kiddo2)

    #This funciton I found somewhere on StackOverFlow. The df parts at least.
    #Basically, we pass in an eliteSize, so like "Top 3" or "Top 5"
    #So the Top 5 populations, or maps, are kept intact. Meaning they survive for sure.
    #The remaining 5 are selected randomly, could be repeated. So we can have repeated maps. This doesn't matter because
    #We don't really save them anywhere (yet). And since we mutate and stuff it'll be different anyway.
    #And it helps because then we wont "run" out of maps.
    #I'm not sure if we NEED it, but I am using it for now to see where I go from here. Maybe it helps in recognizing the "best" map
    def perform_selection(self, pop, eliteSize):
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
                
        return selected_values

    '''def get_following_gen(existing_gen, eliteSize, mutat_rate):
        pop = rankPathes(existing_gen)
        
        selected_values = perform_selection(pop, eliteSize)
       
        my_mating_pool = do_mating_pool(existing_gen, selected_values)
        tot = do_breed_population(my_mating_pool, eliteSize)
        following_gen = do_mutatation(tot, mutat_rate)
        #print(following_gen)
        return following_gen
    get_following_gen(population, 5, 0.01)'''

if __name__ == '__main__':
    f = open("TSM.txt", 'r').read().splitlines()
    #List of cities essentially
    cityCoords = np.array([ tuple( map( float, coord.split() ) ) for coord in f ]).tolist()
    city_names = open("city_names.txt", 'r').read().splitlines()
    popSize = 10
    
    #Call the class object
    initial = initialize(cityCoords, city_names, popSize)
    
    listOfPaths= initial.generatePath()
    #distances = initial.distance_between_cities(cityCoords).values()
    
    population = initial.initialPopulation()
    
    for idx, pop_plot in enumerate(population):
        print('Init pop. ' + str(idx), pop_plot)
        print('\n')
        initial.plot_pop(pop_plot)

    genetics = GeneticAlgorithm()
    fitness = genetics.path_fitness(cityCoords)
    ranks = genetics.rankPaths(population)
    #selected_values = genetics.perform_selection(ranks, 5)
    
    elites = 5

    #popp = genetics.rankPaths(population)
    selected_values = genetics.perform_selection(ranks,elites)
    
    poolForSex = genetics.get_pool_for_sex(population, selected_values)
    
    after = genetics.population_after_sex(poolForSex, elites)
    print("-----------------------------BEFORE SEX-----------------------")
    print(poolForSex)

    print("-----------------------------AFTER SEX-----------------------")
    print(after)
    print("-----------------------------AFTER MUTATION-----------------------")

    #Now we make them mate
    #genetics.haveSex(poolForSex, elites)
    
    #Can use f and s for poolForSex since already in maps form.
    #print(selected_values)
    #print(poolForSex)
    



    #new_pop = genetics.mating(ranks)
    #print(new_pop)
    
    
#I still gotta fix the functions to make sure they all work fine in the class, but it works.
#Now we just need to do the Genetic Algorithm stuff. We can do that in the class.
#After the fitness, we need to rank them, select, and mate/breed the bitches.
