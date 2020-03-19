#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

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
        print(content)
        
    city_names = content
    
    
    plt.figure(figsize=(20,10))
    x = [i[0] for i in cities]
    y = [i[1] for i in cities]
    x1=[x[0],x[-1]]
    y1=[y[0],y[-1]]
    plt.plot(x, y, 'b', x1, y1, 'b')
    plt.scatter (x, y)
    
    x_vals = []
    y_vals = []
    
    for i in range(len(cities)):
        x_vals.append(cities[i][0])
        y_vals.append(cities[i][1])
    
    for i, txt in enumerate(city_names):
        plt.annotate(txt, (x_vals[i], y_vals[i]),horizontalalignment='center', 
            #verticalalignment='bottom',
                    )
    plt.show()
    return

def initialPopulation(cities, populationSize):
    population = [generatePath(cities) for i in range(0, populationSize)]
    return population


# In[3]:


f = open("TSM.txt", 'r').read().splitlines()
numCities = f.pop(0)
cities = np.array([ tuple( map( float, coord.split() ) ) for coord in f ]).tolist()

val = distance_between_cities(cities)

print(val)

tot = total_distance(cities)

print("total distance: ", tot)
list= generatePath(cities)

population = initialPopulation(cities,10)


# In[6]:


for idx, pop_plot in enumerate (population):
    print('Initial Population '+ str(idx),pop_plot)
    plot_pop(pop_plot)

#for pop_plot in population:
 #   plot_pop(pop_plot)


# In[ ]:





# In[ ]:




