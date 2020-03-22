#!/usr/bin/env python
# coding: utf-8

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

<<<<<<< HEAD
def distance_between_cities(cities):
    data = dict()
    for index, value in enumerate(cities):
        x1 = cities[index][0]
        print(x1, "x1")
        y1 = cities[index][1]
        print(y1, "y1")
        print(index+1)
        print(len(cities) -1)
        if index + 1 <= len(cities)-1:
            x2 = cities[index+1][0]
            print(x2, "x2")
            y2 = cities[index+1][1]
            print(y2, "y2")
            xdiff = x2 - x1
            ydiff = y2 - y1
            dst = (xdiff*xdiff + ydiff*ydiff)** 0.5 #The distance formula
            data['Distance from city '+ str(index+1) +' to city ' + str(index+2)] = dst 
        elif index + 1 > len(cities)-1: #Need or it will go out of bounds
            x2 = cities[0][0]
            y2 = cities[0][1]
            xdiff = x2 - x1
            ydiff = y2 - y1
            dst = (xdiff*xdiff + ydiff*ydiff)** 0.5 #The distance formula
            data['Distance from city '+ str(index+1) + ' to city ' + str(index +2 -len(cities))] = dst
              
    return data

def total_distance(cities):
    total = sum(distance_between_cities(cities).values())
    return total

#The random.sample() returns a list of unique elements chosen randomly from the list, sequence, or set, 
#we call it random sampling without replacement. In simple terms, for example, you have a list of 100 names, 
#and you want to choose ten names randomly from it without repeating names, then you must use random.sample().

def generatePath(cities):
    path = random.sample(cities, len(cities))
    return path

def plot_pop(cities):
    with open("city_names.txt") as f:
        content = f.readlines()
    content = list(map(lambda s: s.strip(), content))    
       
    city_names = content
   
    plt.figure(figsize=(20,10))
    x = [i[0] for i in cities]
    y = [i[1] for i in cities]
    x1=[x[0],x[-1]]
    y1=[y[0],y[-1]]
    plt.plot(x, y, 'b', x1, y1, 'b')
    plt.scatter (x, y)
=======
class initialize:
>>>>>>> ffb02d5c3d034e68ed77997748da3b4591270c2c
    
    def __init__(self, cities, city_names, populationSize):
        self.cities = cities
        self.city_names = city_names
        self.populationSize = populationSize
        #self.cityList = cityList
    
<<<<<<< HEAD
    for i in range(len(cities)):
        x_vals.append(cities[i][0])
        y_vals.append(cities[i][1])

    for i, txt in enumerate(city_names):
        plt.annotate(txt, (x_vals[i], y_vals[i]),horizontalalignment='center', 
            #verticalalignment='bottom',
                    )
    plt.show()
    
    # combines city names and coordinates
    comboCoords = list(zip(content, x_vals, y_vals))
    return comboCoords

def generateChildren(coords):
		# shuffles zipped list of city names and coordinates
		# returns shuffled list
		new = random.shuffle(coords)
		return new
# In[71]:


f = open("TSM.txt", 'r').read().splitlines()
numCities = f.pop(0)
# cities is a list of all of the coordinates 
cities = np.array([ tuple( map( float, coord.split() ) ) for coord in f ]).tolist()
print(cities, "cities")
val = distance_between_cities(cities)

print(val, "dist btwn cities")

tot = total_distance(cities)

print("total distance: ", tot)

# list is the (x,y) coordinate pairs for all 30 cities
city_coords= generatePath(cities)

# plot cities
comboCoords1 = plot_pop(city_coords)
# shuffles list, which generates 1 new child
# could put FOR loop into generateChildren() to have it make
# multiple kids
newKids = generateChildren(comboCoords1)
=======


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
>>>>>>> ffb02d5c3d034e68ed77997748da3b4591270c2c


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
        print(gen1Length, 'Gen1')
        gen2Length = int(random.random() * len(mother))
        print(gen2Length, 'GEN2')
        first_generation = min(gen1Length, gen2Length)       
        last_generation = max(gen1Length, gen2Length)
        
        print(father, "FAATHER")
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
        print(mother, "MOOTHER")
        print(tot_parent1)
        #Whatever isn't from father, include to here
        tot_parent2 = [i for i in mother if i not in tot_parent1]
        print(tot_parent2)
        
        #Returns the combination of the two for ONE MAP. For i=2 in range(LosingLength), new values for gen1length and gen2length are formed and the same thing is done.
        #In the end, we should have run this 5 times if the elites was 5. Or 9 times if the elites was 1.
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
        print("-------------POPULATION AFTER SEX FUNCTION----------------")
        #This just randomizes order of the pool and makes Good the pool's length
        GoodLength = random.sample(poolForSex, len(poolForSex))
        print(GoodLength, "GOOD LENGTH")
        print("POOL FOR SEX", poolForSex)
        #This is the first part of the population of only the elites. So need part 2 of the people to have sex to survive
        #This hold 5 maps, NOT 5 cities, depending on the elites value. If elites was 2, part1 hold 2 MAPS.
        part1 = [poolForSex[i] for i in range(elites)]
        print("PART 1!!!!!!!!!!!", part1)
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
    
    #Need to fixto work with functions.
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
            
        print(selected_values, '!st selection!!')
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
    
    #This is the number of "fit" people per generation. Can change it
    elites = 5

    #popp = genetics.rankPaths(population)
    
    #This returns the top 5 values AND selections of any of the people in the rankings.
    #So returns the values likes [0,2, 3,4,5,6,4,5,3,1]
    #Where the first 5 in the list are the top 5 rankings of the population (look at ranks)
    #Andd the rest are selections of ANY remaining 5, can be repeated but it doesn't matter because we will mate
    selected_values = genetics.perform_selection(ranks,elites)
    
    #This returns the population pool that will be used to mate using the selected values
    poolForSex = genetics.get_pool_for_sex(population, selected_values)
    
    #The population after having sex
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
    

    #Need to do: Fix mutation and make generate_next_population() using all the functions.
