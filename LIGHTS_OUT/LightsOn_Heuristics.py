#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:52:07 2019

@author: german
"""

import numpy as np
import pdb
import random
import collections
from queue import PriorityQueue
import heapq
import queue as Q
import operator
import math
import time

L = []
L_dict = {}
Lseen = []
actionlist = []
myL = []
L_tier = {}
all_childs = []
all_actions = []
childs_2 = []
actions_2 = []
next_L_tier = {}


class SearchAI:
    def __init__(self, Istate, goal_state, numRows, numColumns):
        self.Istate = Istate
        self.numRows = numRows
        self.numColumns = numColumns
        self.goal_state = goal_state

    # This function finds for the child states
    def toggle(self, parent_state, numRows, numColumns):
        temp_state = np.copy(parent_state)
        child_states = []
        action = list()
        for i in range(numRows):
            for j in range(numColumns):
                temp_state[i, :] = np.logical_not(parent_state[i, :])
                # pdb.set_trace()
                temp_state[:, j] = np.logical_not(parent_state[:, j])
                # pdb.set_trace()
                child_states.append(temp_state)
                temp_state = np.copy(parent_state)
                action.append([i, j])
        return child_states, action

    # convert String
    def convert2str(self, array, numRows, numColumns):
        value = ""
        for i in range(numRows):
            for j in range(numColumns):
                value = value + str(array[i, j])
        # print(value, 'value from get state!!!!')
        return value

    def getArray(self, state, numRows, numColumns):
        array = np.zeros((numRows, numColumns),  dtype=int)
        for i in range(numRows):
            for j in range(numColumns):
                array[i, j] = state[i * numColumns + j]
        return array

    def sumStr(self, string):
        summ = 0
        for letter in string:
            summ = summ + int(letter)
        return summ

    def evaluate(self, currentArray, goal, numRows, numColumns, algorithm):
        # Lseen holds the currentArray. For initial move, currentArray = initial state.
        # After initial move, currentArray = best child.
        # best_child = ''
        Lseen.append(self.convert2str(currentArray, numRows, numColumns))
        # print(Lseen, 'L Seen')
        # print(len(Lseen), 'number of moves made')

        # L_dict tracks ALL unseen children at all tiers. Useful for best first
        # print(L_dict, "L_dict AT START EVAL")

        # GENERATE NEW CHILDREN AND ACTIONS
        childs, action = self.toggle(currentArray, numRows, numColumns)

        # Convert all children to strings
        for y in range(len(childs)):
            childs[y] = self.convert2str(childs[y], numRows, numColumns)

        # NEEDED FOR BFS
        all_childs.append(childs)
        all_actions.append(action)
        # All children and actions appended WITHOUT NESTING
        flat_all_childs = [
            item for sublist in all_childs for item in sublist]
        flat_all_actions = [
            item for sublist in all_actions for item in sublist]
        # for each child....
        for j in childs:
            # if child is not in Lseen...
            if j not in Lseen:
                # if child is not the goal...
                if j == goal:
                    # print("DONE IN EVAL FXT")
                    print('L Seen: ', Lseen)
                    print('number of moves: ', len(Lseen))
                    Lseen.clear()
                    L_dict.clear()
                    L_tier.clear()
                    childs_2.clear()
                    actions_2.clear()
                    return goal, goal

                # print(j, "JJJJJJJJJJJJJJJJJJJJJJJJJ")
                # Sum all ones in child
                summa = self.sumStr(j)

                # UNSEEN CHILDREN IN TIER
                L_tier.update({j: summa})

                # L DICT = ALL UNSEEN CHILDREN
                L_dict.update({j: summa})
                # print(L_tier, "LTIERLTIERLTIER")
                # print(L_dict, "LDICTLDICTLDICT")

                # beginning of look ahead function...
                # next_childs, next_action = self.toggle(self.getArray(
                #     j, numColumns, numColumns), numRows, numColumns)
                # # max dictionary value
                # for k in next_childs:
                #     k = self.convert2str(k, numRows, numColumns)
                #     summb = self.sumStr(k)
                #     next_L_tier.update({k: summb})
                # max_tier_value = max(next_L_tier.values())
                # L_tier.update({j: max_tier_value})
                # L_dict.update({j: max_tier_value})
                # ...end of look ahead function

            # print(L_tier, "UNSEEN CHILDREN IN THE LEVEL")
            # print(L_dict, 'ALL UNSEEN CHILDREN')

        # if algorithm Hill Climbing
        if algorithm == "Hill":
            # Look at max value in dict
            max_value = max(L_tier.values())  # max dictionary value
            # Find key for max value
            max_keys = [k for k, v in L_tier.items() if v == max_value]

            # IF MORE THAN ONE KEY, pick key randomly...
            if len(max_keys) == 1:
                best_child = max_keys[0]
                # print(best_child, "BEST CHILD ONLY CHILD")
            else:
                best_child = random.choice(max_keys)
                # print(best_child, "BEST CHILD CHOSEN RANDOMLY")
            # gets index in childs list of best child
            # uses that index to select the correct action to add to actionlist
            # print(childs, "CHILDSSSSSSSSS")
            # print(action, "ACTIONNSSSSSSSS")
            index = childs.index(best_child)
            actionlist.append(action[index])

        # if algorithm BFS
        if algorithm == "BFS":
            # Look at max value in dict
            max_value = max(L_dict.values())  # max dictionary value
            # Find key for max value
            max_keys = [k for k, v in L_dict.items() if v == max_value]

            # IF MORE THAN ONE KEY, pick key randomly...
            if len(max_keys) == 1:
                best_child = max_keys[0]
                # print(best_child, "BEST CHILD")
            else:
                best_child = random.choice(max_keys)
                # print(best_child, "BEST CHILD CHOSEN RANDOMLY")
            # gets index in childs list of best child
            # uses that index to select the correct action to add to actionlist
            index = flat_all_childs.index(best_child)
            actionlist.append(flat_all_actions[index])

        # print(best_child, "Best Child EVAL")
        # deletes best child from L_dict
        del L_dict[best_child]

        # clears L_tier for each TIER (for Hill Climbing)
        L_tier.clear()
        next_L_tier.clear()
        return best_child, actionlist

    def hillClimbing(self, Istate, goal, numRows, numColumns):
        print("**************************************************************")
        print('-------------------- HILL CLIMBING ------------------------')
        # print('-------------------- HILL CLIMBING ------------------------')
        # print('-------------------- HILL CLIMBING ------------------------')
        # print('-------------------- HILL CLIMBING ------------------------')
        print("**************************************************************")
        # Lseen = []
        # set currentArray equal to initial state
        currentArray = self.getArray(Istate, numRows, numColumns)
        # set flag = 0 for while loop
        flag = 0

        while flag == 0:
                # evaluate returns best child AND the list of current selected actions
            best_child, actionlist = self.evaluate(
                currentArray, goal, numRows, numColumns, "Hill")

            # update currentArray wit best child
            currentArray = self.getArray(best_child, numRows, numColumns)
            # print("CHOSEN ACTIONS: ", actionlist)
            # print('BEST CHILDREN: ', best_child)
            # print('-------------------------------')
            # print('-------------------------------')
            # print('-------------------------------')
            # if we reach the goal, END
            if best_child == goal:
                break

    def best_first_search(self, Istate, goal, numRows, numColumns):
        print("**************************************************************")
        print('-------------------- START BEST FIRST ------------------------')
        # print('-------------------- START BEST FIRST ------------------------')
        # print('-------------------- START BEST FIRST ------------------------')
        # print('-------------------- START BEST FIRST ------------------------')
        print("**************************************************************")
        # set currentArray equal to initial state
        currentArray = self.getArray(Istate, numRows, numColumns)
        # set flag = 0 for while loop
        flag = 0

        while flag == 0:
            # evaluate returns best child AND the list of current selected actions
            best_child, actionlist = self.evaluate(
                currentArray, goal, numRows, numColumns, "BFS")

            # update currentArray wit best child
            currentArray = self.getArray(best_child, numRows, numColumns)
            # print("CHOSEN ACTIONS: ", actionlist)
            # print('BEST CHILDREN: ', best_child)
            # print('-------------------------------')
            # print('-------------------------------')
            # print('-------------------------------')
            # if we reach the goal, END
            if best_child == goal:
                break


if __name__ == '__main__':
        # From the YT Video. Should have a solution but we get errors??
        #start = '110100010'
        #goal = '000000000'
    start = '0000000000000000'
    goal = '1111111111111111'
    numRows = 4  # int(math.sqrt(len(start)))
    numColumns = 4  # numRows
    start_time = time.time()
    SearchAI = SearchAI(start, goal, numRows, numColumns)
    SearchAI.hillClimbing(start, goal, numRows, numColumns)
    # SearchAI.best_first_search(start, goal, numRows, numColumns)
    print("--- %s seconds ---" % (time.time() - start_time))
