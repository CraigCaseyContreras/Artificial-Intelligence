#!/usr/bin/env python
# coding: utf-8

# In[68]:


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
    content = list(map(lambda s: s.strip(), content))    
       
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


