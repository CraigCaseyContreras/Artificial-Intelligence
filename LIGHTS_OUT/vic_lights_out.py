import numpy as np
import random

class SearchAI:
    def __init__(self, Istate, goal_state, numRows, numCols):
        self.Istate = Istate
        self.numRows = numRows
        self.numCols = numCols
        self.goal_state = goal_state

    ## This function finds for the child states
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

    # get representation for state
    def getState(self, array, numRows, numColumns):
        value = ""
        for i in range(numRows):
            for j in range(numColumns):
                value = value + str(array[i, j])
        return value

    def getArray(self, state, numRows, numColumns):
        array = np.zeros((numRows, numColumns), dtype=int)
        for i in range(numRows):
            for j in range(numColumns):
                array[i, j] = state[i * numColumns + j]
        return array

    def printState(self, state, numRows, numColumns):
        for i in range(numRows):
            row = ""
            for j in range(numColumns):
                index = i * numColumns + j
                row = row + " " + str(state[index])
            print(row)

    # Produce a backtrace of the actions taken to find the goal node, using the
    # recorded meta dictionary
    def construct_path(self, state, meta):
        action_list = list()
        # Continue until you reach root meta data (i.e. (None, None))
        # pdb.set_trace()
        while meta[state][0] is not None:
            state, action = meta[state]
            action_list.append(action)
        action_list.reverse()
        return action_list

    def printWinningSeqState(self, Istate, actions, numRows, numColumns):
        # winningList = list()
        self.printState(Istate, numRows, numColumns)
        state = Istate
        for idx in actions:
            # debug
            # winningList.append(state)
            ####
            array = self.getArray(state, numRows, numColumns)
            tempArray = np.copy(array)
            tempArray[idx[0], :] = np.logical_not(array[idx[0], :])
            tempArray[:, idx[1]] = np.logical_not(array[:, idx[1]])
            state = self.getState(tempArray, numRows, numColumns)
            print('---------')
            self.printState(state, numRows, numColumns)

    # def breadth_search(self, Istate=None, goal_state=None, numRows=None, numColumns=None):
    def breadth_search(self, Istate, goal_state, numRows, numColumns):
        # if Istate == None:
        #     Istate = self.Istate
        # if numRows == None:
        #     numRows = self.numRows
        # if numColumns == None:
        #     numColumns = self.numCols
        # if goal_state == None:
        #     goal_state = self.goal_state

        ## 2-Create Lseen and L
        Lseen = set()
        L = list()

        ## 3-Find childrens
        # Istate = getState(np.zeros((numRows, numColumns),  dtype=int), numRows, numColumns)
        # goal_state = getState(np.ones((numRows,numColumns), dtype=int), numRows, numColumns)
        # goal_state = '01010101010101010101'
        print("goal state is")
        print(goal_state)
        L.append(Istate)

        # a dictionary to maintain meta information (used for path formation)
        # key -> (parent state, action to reach child)
        meta = dict()
        meta[Istate] = (None, None)
        # breadth first search
        # 1.Let L be list of initial Nodes
        # 2. let n be first node in L, if L is empty set, fail
        # 3. if n is terminal state, stop
        # 4. remove n from L, add to end of L all of n's childrens randomly ordered
        # 5. Go to 2
        while len(L) > 0:
            currentState = L[0]
            currentArray = self.getArray(currentState, numRows, numColumns)

            # print the state
            # print("****************************")
            # print("Current State is")
            # self.printState(currentState, numRows, numColumns)
            # print("****************************")

            if currentState == goal_state:
                print('Reach goal state')
                print("****************************")
                print('Winning sequence of actions:')
                action_list = self.construct_path(currentState, meta)
                print(action_list)
                self.printWinningSeqState(Istate, action_list, numRows, numColumns)
                break
            else:
                if currentState not in Lseen:
                    Lseen.add(currentState)
                    # print("Length of Lseen", len(Lseen))
                    childs, action = self.toggle(currentArray, numRows, numColumns)
                    c = list(zip(childs, action))
                    random.shuffle(c)
                    childs, action = zip(*c)
                    # random.shuffle(childs)
                    for i in range(len(childs)):
                        state = self.getState(childs[i], numRows, numColumns)
                        notInL = state not in L
                        notInLseen = state not in Lseen
                        if notInL and notInLseen:
                            L.append(state)
                            meta[state] = (currentState, action[i])  # create metadata for these nodes
                L.remove(currentState)
        return action_list


def main():
    # numRows = 5
    # numCols = 5
    # if len(sys.argv) > 2:
    #     numRows = int(sys.argv[1])
    #     numCols = int(sys.argv[2])
    # else:
    SearchAI('0000', '1111', 2, 2)
    SearchAI.breadth_search()


if __name__ == '__main__':
    main()