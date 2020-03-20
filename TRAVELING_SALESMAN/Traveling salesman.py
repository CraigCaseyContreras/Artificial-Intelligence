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
    def rankPathes(self, population):
        results = {}
        for i in range(len(population)):
            results[i] = self.path_fitness(population[i])
            
        #This just sorts them in order from greated fitness to least fitness
        # population #: fitness value. There are 10 in total because made 10 populations
        return sorted(results.items(), key = operator.itemgetter(1), reverse = True)

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
    ranks = genetics.rankPathes(population)
    
#I still gotta fix the functions to make sure they all work fine in the class, but it works.
#Now we just need to do the Genetic Algorithm stuff. We can do that in the class.
#After the fitness, we need to rank them, select, and mate/breed the bitches.
