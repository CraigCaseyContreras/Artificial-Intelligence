#!/usr/bin/env python
# coding: utf-8

# In[190]:


import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

class initialize:
    
    def __init__(self, cities, city_names, populationSize):
        self.cities = cities
        self.city_name = city_names
        self.populationSize = populationSize
        #self.cityList = cityList
    
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
        total = sum(distance_between_cities(cities).values())
        return total

    def generatePath(self):
        path = random.sample(self.cities, len(self.cities))
        return path

    def plot_pop(self, cities, city_names, cityList):
        xList = []
        yList = []
        for i in range(len(cityList)):
            xList.append(cityList[i][0])
            yList.append(cityList[i][1])

        plt.figure(figsize=(20,10))
        x = [i[0] for i in cities]
        y = [i[1] for i in cities]
        x1=[x[0],x[-1]]
        y1=[y[0],y[-1]]
        plt.plot(x, y, 'b', x1, y1, 'b')
        plt.scatter (x, y) 

        for i, txt in enumerate(city_names):
            plt.annotate(txt, (xList[i], yList[i]),horizontalalignment='center', 
                verticalalignment='bottom',
                        )
            print(txt, xList[i], yList[i])
        plt.show()
        return

    def initialPopulation(self):
        population = [generatePath(self.cities) for i in range(0, self.populationSize)]
        return population


class GeneticAlgorithm:
    def __init(self):
        return
    
    #For the fitness levels, I base it off of total distance. We can decide on this, but I get the total distances and 
    #do 1/totalDistance.
    def path_fitness(self, cities):
        total_dis = total_distance(cities)
        fitness = 1 / float(total_dis)
        return fitness

if __name__ == '__main__':
    f = open("TSM.txt", 'r').read().splitlines()
    cityCoords = np.array([ tuple( map( float, coord.split() ) ) for coord in f ]).tolist()
    #print("City Coords: ", cityCoords)
    #val = distance_between_cities(cityCoords).values()
    list= generatePath()
    #print(list)
    city_names = open("city_names.txt", 'r').read().splitlines()
    
    initial = initialize(cityCoords, city_names, 10)
    
    population = initial.initialPopulation()
    for idx, pop_plot in enumerate(population):
        print('Init pop. ' + str(idx), pop_plot)
        print('\n')
        initial.plot_pop(pop_plot, city_names, cityCoords)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




