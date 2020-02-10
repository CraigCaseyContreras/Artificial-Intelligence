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

class SearchAI:
    def __init__(self,Istate,goal_state, numRows, numCols):
        self.Istate=Istate
        self.numRows = numRows
        self.numCols = numCols
        self.goal_state = goal_state
    ## This function finds for the child states
    def toggle(self,parent_state, numRows, numColumns):
        temp_state = np.copy(parent_state)
        child_states = []
        action = list()
        for i in range(numRows):
            for j in range(numColumns):
                    temp_state[i,:] = np.logical_not(parent_state[i,:]) 
                    #pdb.set_trace()
                    temp_state[:,j] = np.logical_not(parent_state[:,j]) 
                    #pdb.set_trace()
                    child_states.append(temp_state)
                    temp_state = np.copy(parent_state)
                    action.append([i,j])
        return child_states, action
    
    # get representation for state
    def getState(self,array, numRows, numColumns):
        value = ""
        for i in range(numRows):
            for j in range(numColumns):
                value = value + str(array[i,j])
        return value
        
    def getArray(self,state, numRows, numColumns):
        array = np.zeros((numRows, numColumns),  dtype=int)
        for i in range(numRows):
            for j in range(numColumns):
                array[i,j] = state[i * numColumns + j]
        return array
    
    def printState(self,state, numRows, numColumns):
        for i in range(numRows):
            row = ""
            for j in range(numColumns):
                index = i * numColumns + j
                row = row + " " + str(state[index])
            print(row)
    # Produce a backtrace of the actions taken to find the goal node, using the 
    # recorded meta dictionary
    def construct_path(self,state, meta):
        action_list = list()
        # Continue until you reach root meta data (i.e. (None, None))
        #pdb.set_trace()
        while meta[state][0] is not None:
            state, action = meta[state]
            action_list.append(action)
        action_list.reverse()
        return action_list
    def printWinningSeqState(self,Istate,actions,numRows, numColumns):
            #winningList = list()
            self.printState(Istate,numRows, numColumns)
            state=Istate
            for idx in actions:
                    # debug
                    #winningList.append(state)
                    ####
                    array=self.getArray(state, numRows, numColumns)
                    tempArray = np.copy(array)
                    tempArray[idx[0],:] = np.logical_not(array[idx[0],:]) 
                    tempArray[:,idx[1]] = np.logical_not(array[:,idx[1]])
                    state = self.getState(tempArray, numRows, numColumns)
                    print('---------')
                    self.printState(state,numRows, numColumns)     
    def breadth_search(self,Istate=None,goal_state=None,numRows=None,numColumns=None):
        if Istate == None:
            Istate = self.Istate
        if numRows == None:
            numRows = self.numRows
        if numColumns == None:
            numColumns = self.numCols
        if goal_state == None:
            goal_state = self.goal_state
        ## 2-Create Lseen and L
        Lseen = set()
        L = list()
        ## 3-Find childrens
        #Istate = getState(np.zeros((numRows, numColumns),  dtype=int), numRows, numColumns)
        #goal_state = getState(np.ones((numRows,numColumns), dtype=int), numRows, numColumns)
        #goal_state = '01010101010101010101'
        print("goal state is")
        print(goal_state)
        L.append(Istate)
        
        # a dictionary to maintain meta information (used for path formation)
          # key -> (parent state, action to reach child)
        meta = dict()
        meta[Istate] = (None,None)
        #breadth first search
        #1.Let L be list of initial Nodes
        #2. let n be first node in L, if L is empty set, fail
        #3. if n is terminal state, stop
        #4. remove n from L, add to end of L all of n's childrens randomly ordered
        #5. Go to 2
        while len(L) > 0:
            currentState = L[0]
            currentArray = self.getArray(currentState, numRows, numColumns)
            
            #print the state
            #print("****************************")
            #print("Current State is")
            #self.printState(currentState, numRows, numColumns)
            #print("****************************")
            
            if currentState == goal_state:
                print('Reach goal state') 
                print("****************************")
                print('Winning sequence of actions:')
                action_list=self.construct_path(currentState, meta)
                print(action_list)
                self.printWinningSeqState(Istate,action_list,numRows, numColumns)
                break
            else: 
                if currentState not in Lseen:
                    Lseen.add(currentState)
                    #print("Length of Lseen", len(Lseen))
                    childs, action = self.toggle(currentArray, numRows, numColumns)
                    c = list(zip(childs, action))
                    random.shuffle(c)
                    childs, action = zip(*c)
                    #random.shuffle(childs)
                    for i in range(len(childs)):
                        state = self.getState(childs[i], numRows, numColumns)
                        notInL = state not in L
                        notInLseen = state not in Lseen
                        if notInL and notInLseen:
                            L.append(state)
                            meta[state] = (currentState, action[i]) # create metadata for these nodes
                L.remove(currentState) 
        return action_list
    
    ## Depth first function
    def depth_search(self,Istate=None,goal_state=None,numRows=None,numColumns=None):
        if Istate == None:
            Istate = self.Istate
        if numRows == None:
            numRows = self.numRows
        if numColumns == None:
            numColumns = self.numCols
        if goal_state == None:
            goal_state = self.goal_state
        ## 2-Create Lseen and L
        Lseen = set()
        L = list()
        ## 3-Find childrens
        #numRows = 4
        #numColumns = 4
        #Istate = getState(np.zeros((numRows, numColumns),  dtype=int), numRows, numColumns)
        #goal_state = getState(np.ones((numRows,numColumns), dtype=int), numRows, numColumns)
        print("goal state is")
        print(goal_state)
        L.append(Istate)
        
        # a dictionary to maintain meta information (used for path formation)
          # key -> (parent state, action to reach child)
        meta = dict()
        meta[Istate] = (None,None)
        #depth first search
        #1.Let L be list of initial Nodes
        #2. let n be first node in L, if L is empty set, fail
        #3. if n is terminal state, stop
        #4. remove n from L, add to front of L all of n's childrens randomly ordered
        #5. Go to 2
        
        Lseen = set()
        L = list()
        
        L.append(Istate)
        while len(L) > 0:
            currentState = L[0]
            currentArray = self.getArray(currentState, numRows, numColumns)
            
            #print the state
            #print("****************************")
            #print("Current State is")
            #printState(currentState, numRows, numColumns)
            #print("****************************")
            
            if currentState == goal_state:
                print('Reach goal state') 
                print('Winning sequence of actions:')
                action_list=self.construct_path(currentState, meta)
                print(action_list)
                self.printWinningSeqState(Istate,action_list,numRows, numColumns)
                break
            else: 
                if currentState not in Lseen:
                    Lseen.add(currentState)
                    #print("Length of Lseen", len(Lseen))
                    childs, action = self.toggle(currentArray, numRows, numColumns)
                    c = list(zip(childs, action))
                    random.shuffle(c)
                    childs, action = zip(*c)
                    #random.shuffle(childs)
                    #for array in childs:
                    for i in range(len(childs)):
                        #state = getState(array, numRows, numColumns)
                        state = self.getState(childs[i], numRows, numColumns)
                        notInL = state not in L
                        notInLseen = state not in Lseen
                        if notInL and notInLseen:
                            L.insert(0, state)
                            meta[state] = (currentState, action[i]) # create metadata for these nodes
                L.remove(currentState)
        return action_list 
    
    # Iterative deepening function
    def iterative_deep_search(self,Istate=None,goal_state=None,numRows=None,numColumns=None):
        if Istate == None:
            Istate = self.Istate
        if numRows == None:
            numRows = self.numRows
        if numColumns == None:
            numColumns = self.numCols
        if goal_state == None:
            goal_state = self.goal_state
        #numRows = 5
        #numColumns = 5
        #Istate = getState(np.zeros((numRows, numColumns),  dtype=int), numRows, numColumns)
        #Istate = '011101'
        #goal_state = getState(np.ones((numRows,numColumns), dtype=int), numRows, numColumns)
        #goal_state = '01010101010101010101'
        print("goal state is")
        print(goal_state)
        
        # a dictionary to maintain meta information (used for path formation)
          # key -> (parent state, action to reach child)
        meta = dict()
        meta[Istate] = (None,None)
        
        #ITERATIVE DEEPENING
        #for i = 1,..., X do depth first w/ depth i
        
        #1. let c = 1 be current depth cutoff
        #2. Let L be list of initial nodes
        #3. Let n be first node in L, if L is empty, set c = c + 1 and return to step 2
        #4. if n is terminal state, stop
        #5. remove n from L, if dn < c, add to front of L, all of n's children, go to 3
        
        depth = 1
        
        Lseen = list()
        L = list()
        Ldepth = list()
        L.append(Istate)
        Ldepth.append(0)
        flag = 0
        while True:
            currentState = L[0]
            currentDepth = Ldepth[0]
            #currentState = getStateDeep(current)
            #currentDepth = getDepth(current)
            currentArray = self.getArray(currentState, numRows, numColumns)
            
            #print the state
            #print("****************************")
            #print("Current State is")
            #printState(currentState, numRows, numColumns)
            #print("****************************")
        
            if currentState == goal_state:
                Lseen.insert(0,currentState)
                print('Reach goal state') 
                print('Winning sequence of actions:')
                action_list=self.construct_path(currentState, meta)
                print(action_list)
                self.printWinningSeqState(Istate,action_list,numRows, numColumns)
                break
            
            if currentState not in Lseen:
                #pdb.set_trace()
                #Lseen.add(currentState)
                Lseen.insert(0,currentState)
                L.pop(0) # remove firt n node from L and its corresponding depth
                Ldepth.pop(0)
                #print("Length of Lseen is: ", len(Lseen)) 
                if currentDepth + 1 < depth:
                    childs, action = self.toggle(currentArray, numRows, numColumns)
                    #pdb.set_trace()
                    #for array in childs:
                    for i in range(len(childs)):
                        state = self.getState(childs[i], numRows, numColumns)
                        stateDepth = currentDepth + 1
                        notInL = state not in L
                        notInLseen = state not in Lseen
                        if notInL and notInLseen:
                            L.insert(0, state)
                            Ldepth.insert(0,stateDepth)
                            meta[state] = (Lseen[0], action[i]) # create metadata for these nodes
            #L.remove(current)
            if len(L) == 0:
                Lseen = list()
                L.append(Istate)
                Ldepth.append(0)
                depth = depth + 1
                print("depth is ", depth)
        return action_list