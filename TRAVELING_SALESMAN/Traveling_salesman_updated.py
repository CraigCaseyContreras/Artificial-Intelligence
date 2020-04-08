#!/usr/bin/env python
# coding: utf-8

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

count = 0
stdev_list = []
mean_list = [] 

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
		print('\n')
		plt.show()
		return

	def initialPopulation(self):
		population = [self.generatePath() for i in range(0, self.populationSize)]
		return population


class GeneticAlgorithm:
    
	#def __init__(self, cities):
		#self.cities = cities
    
	def counter(self):
			global count
			count = count + 1
			return count
    
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
		fitness = float(total_dis)
		#fitness = 1 / float(total_dis)
		return fitness
		
	#Rank them bitches
	
	def rankPaths(self, population):
		#results = []
		results = {}

		resultor = list()
		count = self.counter()
		for i in range(len(population)):
				results[i] = self.path_fitness(population[i])
        
		resultor = list(results.values())
		mean_list.append(np.mean(resultor))
		print(mean_list, "MEANNNNNNNNNNNNNNN LISTTTT")
		std_dev = 0.5*np.std(resultor)
		stdev_list.append(std_dev)
		print(stdev_list, "STDDDDDDDDDDDDDDD")
		if count == 1:
				init_thresh = mean_list + std_dev
				new_dict = {key:val for key, val in results.items() if val < (init_thresh)}
				print("IF THRESH IS ", init_thresh)
		else:
				print(count, "COUNTTTTT")
				thresh = mean_list[0] - np.sum(stdev_list)
				print(thresh, "THRESHHHHH")
				new_dict = {key:val for key, val in results.items() if val < (thresh)}
				print('ELSE THRESHOLD IS ', thresh)    
		return list(new_dict.keys())
    
    
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
		child = tot_parent1 + tot_parent2
     
        
		#Returns the combination of the two for ONE MAP. For i=2 in range(LosingLength), new values for gen1length and gen2length are formed and the same thing is done.
		#In the end, we should have run this 5 times if the elites was 5. Or 9 times if the elites was 1.
		return child
        
        
    
	#MATING STUFF GOES HERE!!!!!
	def population_after_sex(self, poolForSex, elites):
		#print(poolForSex, 'PPPOOOOLLLL')
		pp = []
        
		for w in range(10):
				one_parent = int(random.uniform(0,elites))
				two_parent = int(random.uniform(0,elites))
				if one_parent == two_parent:
					two_parent = int(random.uniform(0,elites))
                
				ppp = self.haveSex(poolForSex[one_parent], poolForSex[two_parent])
				pp.append(ppp)
		pop_after = poolForSex + pp
		return pop_after
    
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
    

	def next_generation(self, population):
		#Population would start with the inital one and then each time after the mutation, it will change
		selections = self.rankPaths(population)
		elites = len(selections)
		rate = random.random()
		poolForSex = self.get_pool_for_sex(population, selections)
		afterSex = self.population_after_sex(poolForSex, elites)
		mutated_pool = self.get_pool_after_mutation(afterSex, rate)
		#print(mutated_pool, "MUTATED")
		return mutated_pool
    
	#Generations meaning how many generations to run the program for
	def pass_time(self, population, generations):
		#Runs the thing generations times
		optimal_routes = []
		for i in range(generations):
				population = self.next_generation(population)
				if len(population) == 1:
						print(population, "OPTIMAL ROUTE")
						print(self.path_fitness(population), "OPTIMAL DIST")
						break
				print("Population",i,population, len(population))
				print('\n')
				best_rank = self.rankPaths(population)

				'''
				optimal_route = population[best_rank[0]]
				optimal_routes.append(optimal_route)
	
	
            
		optimal_dists = []
		for i in range(len(optimal_routes)):
				optimal_dists.append(self.path_fitness(optimal_routes[i]))
            
		#Indices start at 0!!!!!!!!!!!!!!!!
		print('\n')
		print(optimal_routes, 'OPTIMAL ROUTES')
		print('\n')
		print(optimal_dists, 'OPTIMAL DISTS')
		print('\n')
		print('Best route at index',optimal_dists.index(min(optimal_dists)))
		optimal_route_index = optimal_dists.index(min(optimal_dists))
		print(optimal_routes[optimal_route_index], 'BEST ROUTE SO FAR!!!')
		print('\n')
        
		# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		# COMMENTED OUT INITIAL PLOTSSSSSS
		initial.plot_pop(optimal_routes[optimal_route_index])
       '''
		return 


if __name__ == '__main__':

	f = open("TSM.txt", 'r').read().splitlines()
	cityCoords = np.array([ tuple( map( float, coord.split() ) ) for coord in f ]).tolist()
	city_names = open("city_names.txt", 'r').read().splitlines()
	popSize = 10
	initial = initialize(cityCoords, city_names, popSize)
	population = initial.initialPopulation()
   
	# ===================================================
	# COMMENTED OUT INITiAL PLOTSSSSSSSSSSSSSSSSss
	'''
	for idx, pop_plot in enumerate(population):
		print('Init pop. ' + str(idx), pop_plot)
		print('\n')
		initial.plot_pop(pop_plot)
	'''
   
	genetics = GeneticAlgorithm()

	generations = 10
    
	genetics.pass_time(population, generations)
    
	print('-------------------------END----------------------')
