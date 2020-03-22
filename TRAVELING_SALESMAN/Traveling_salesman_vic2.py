#!/usr/bin/env python
# coding: utf-8

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
#=======
class initialize:
  
	def __init__(self, cities, city_names, populationSize):
		self.cities = cities
		self.city_names = city_names
		self.populationSize = populationSize
		x_vals = []
		y_vals = []
		#self.cityList = cityList
    
		for i in range(len(cities)):
			x_vals.append(cities[i][0])
			y_vals.append(cities[i][1])
	
		for i, txt in enumerate(city_names):
			plt.annotate(txt, (x_vals[i], y_vals[i]),horizontalalignment='center', 
					#verticalalignment='bottom',
							)
		plt.show()
		return	
		
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
		# Initializes 10 children. This is our original gene pool. We will breed and mutate
		# this initial population to evolve the algorithm
		population = [self.generatePath() for i in range(0, self.populationSize)]
		return population
"""	
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
print(comboCoords1, "combo coords 1")
"""


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
		results = []
		# NOTE: population is a list of lists. Contains 10 children. Each "child" is a list that orders cities in the
		# order they are visited by the salesman
		for i in range(len(population)):
				results.append(self.path_fitness(population[i]))
		#This just sorts them in order from greatest fitness to least fitness
		# population. There are 10 in total because made 10 populations
		
		# Okay so I made some edits here. Combination is a zipped list that combines the path's fitness levels and path.
		# So the new list will look like [FITNESS_VAL, [PATH LIST]]
		combination = [list(a) for a in zip(results, population)]
		
		return sorted(combination, key = operator.itemgetter(0), reverse = True)
		
	def mating(self, combo1):
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
		kiddo = []
		kiddo2 = []
		
		flag = random.uniform(0,1)
		if flag <= 0.9:
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
		elif flag > 0.9:
			# MUTATION
			# Switches 2 elements in best and 2nd best paths
			f[5],f[6] = f[6],f[5]
			kiddo = f
			s[5],s[6] = s[6],s[5]
			kiddo2 = s
		
		# new_population contains ONLY paths (no 1/distance included)
		# it includes unmated children, and the new child. 
		# I programmed it this way so that mating() returns a new population.
		# This new population can then be plugged into path_fitness() and rankPaths()
		new_population = [n[1:] for n in combo1]
		new_population.append(kiddo)
		new_population.append(kiddo2)
		return new_population

if __name__ == '__main__':
	f = open("TSM.txt", 'r').read().splitlines()
	#List of cities essentially
	cityCoords = np.array([ tuple( map( float, coord.split() ) ) for coord in f ]).tolist()
	print(cityCoords, "City COORDS")
	city_names = open("city_names.txt", 'r').read().splitlines()
	print(city_names, "CITY NAMES")
	popSize = 10
	
   # combines city names and coordinates into 1 list
	comboCoords = list(zip(city_names, cityCoords))
	
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
	# ranks is a nested list containing the [1/distance [and paths]]
	ranks = genetics.rankPaths(population)
	#print(ranks, "RANKS")
	
	# new_pop is a list of all paths. So its another list of lists. 
	# I programmed it this way so that mating() returns a new population.
	# This new population can then be plugged into path_fitness() and rankPaths()
	new_pop = genetics.mating(ranks)
	print(new_pop, "NEW POPULATION")
