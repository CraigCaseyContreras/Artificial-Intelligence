# coding: utf-8

"""
GUI for the Light Out solver.
"""

import sys
import tkinter as tk
from itertools import product

import numpy as np

import lightsearch

BUTTON_SIZE = 50


class ButtonGrid(tk.Frame, object):
    
    def __init__(self, master, numRows, numCols, callback=None):
        tk.Frame.__init__(self, master)
        
        buttons = [[None]*numCols for i in range(numRows)]
        for r,c in product(range(numRows), range(numCols)):
            buttons[r][c] = tk.Label(self, bitmap="gray12", bg="red",
                                width=BUTTON_SIZE, height=BUTTON_SIZE)
            buttons[r][c].grid(row=r, column=c, padx=4, pady=4)
        
        for btn in sum(buttons, []):
            btn.bind("<Button-1>", self.button_pressed)
        
        self.buttons = buttons
        self.callback = callback
        self.numRows = numRows
        self.numCols = numCols
        self.flag_showSolution = False
        self.action_list = list()
        self.sol_length = 0
        self.count = 0
        #self.size = size
    def restart_state(self):
        self.flag_showSolution = False
        self.action_list = list()
        self.sol_length = 0
        self.count = 0
        for r,c in product(range(self.numRows), range(self.numCols)):
            self.buttons[r][c].config(bitmap = "gray12")
            self.buttons[r][c].config(bg = "red")
    def button_pressed(self, event):
        buttonPress = event.widget
        buttonPressInfo = buttonPress.grid_info()
        buttonPressCol = buttonPressInfo["column"]
        buttonPressRow = buttonPressInfo["row"]
        for i in range(self.numCols):# toogle row colors
            if self.buttons[buttonPressRow][i]["bg"] == "red":
                self.buttons[buttonPressRow][i].config(bg = "green")
            else:
                self.buttons[buttonPressRow][i].config(bg = "red")
        for i in range(self.numRows):# toogle col colors
            if self.buttons[i][buttonPressCol]["bg"] == "red":
                self.buttons[i][buttonPressCol].config(bg = "green")
            else:
                self.buttons[i][buttonPressCol].config(bg = "red")
                            
        if buttonPress["bg"] == "red":
            buttonPress["bg"] = "green"
        else:
            buttonPress["bg"] = "red"
        if (self.count < self.sol_length-1) & self.flag_showSolution == True :
            row=self.action_list[self.count][0]
            col=self.action_list[self.count][1]
            self.buttons[row][col].config(bitmap = "gray12")
            self.count = self.count + 1
            row=self.action_list[self.count][0]
            col=self.action_list[self.count][1]
            self.buttons[row][col]["bitmap"] = "gray75"
            
#        if self.callback is not None:
#            self.callback(self.get_board())
    
    def get_board(self):
        # board = map(lambda x:x["bg"]=="green", sum(self.buttons, []))
        board = [i["bg"] == "green" for i in sum(self.buttons, [])]
        return np.reshape(board, (self.numRows, self.numCols))
    
    def set_solution(self, solution):
        if solution is None:
            for btn in sum(self.buttons, []):
                btn["bitmap"] = "gray6"
            return
        self.action_list = solution
        self.sol_length = len(solution)
        self.flag_showSolution = True
        r=solution[0][0]
        c=solution[0][1]
        self.buttons[r][c]["bitmap"] = "gray75"
#        assert solution.shape[0]==solution.shape[1]==self.size
#        
#        for r,c in product(range(self.size), range(self.size)):
#            if solution[r,c]:
#                self.buttons[r][c]["bitmap"] = "gray75"
#            else:
#                self.buttons[r][c]["bitmap"] = "gray12"
    

class App(tk.Frame):
    
    def __init__(self, master, numRows=5,numCols=5):
        tk.Frame.__init__(self, master, width=1000, height=1000)
        self.pack(fill="both")
        
        self.numRows = numRows
        self.numCols = numCols
        #self.solver = lightsout.LightsOut(size)
        
        self.info = tk.Label(self, text="")
        self.info.pack(side=tk.TOP)
        
        self.buttongrid = ButtonGrid(self, self.numRows,self.numCols, callback=self.solve)
        self.buttongrid.pack(padx=10, pady=10)
        
        # insert radio button to select type of search
        self.v = tk.IntVar()
        self.v.set(0)  # initializing the choice, i.e. Python
        search_types = [("DFS",1),("BFS",2),("IDS",3)]
        def ShowChoice():
            print(self.v.get())

        tk.Label(self, text="""Choose type of search:""",justify = tk.LEFT, padx = 20).pack()

        for val, searchType in enumerate(search_types):
            tk.Radiobutton(self, text=searchType,padx = 20, variable=self.v, command=ShowChoice,value=val).pack(anchor=tk.W)
            
        button_search = tk.Button(self, text="Search", fg="black") 
        button_search.pack(side=tk.LEFT)
        button_search.bind("<Button-1>",self.search)
        
        button_restart = tk.Button(self, text="Restart", fg="black") 
        button_restart.pack(side=tk.LEFT)
        button_restart.bind("<Button-1>",self.restart_board)
        
    def restart_board(self,event):
        self.buttongrid.restart_state()
        
    def search(self,event):
        Istate = ""
        Istate_array = self.buttongrid.get_board()
        for i in range(self.numRows):
            for j in range(self.numCols):
                if Istate_array[i,j]==True:
                    char = "1"
                else:
                    char = "0"
                Istate = Istate + char
        goal_state_array = np.ones((self.numRows,self.numCols), dtype=int)
        goal_state = ""
        for i in range(self.numRows):
            for j in range(self.numCols):
                goal_state = goal_state + str(goal_state_array[i,j])
        
        SearchAI = lightsearch.SearchAI(Istate,goal_state,self.numRows,self.numCols)
        self.info["text"] = "Finding Solution"
        self.update_idletasks()
        if self.v.get()==0:
            action_list = SearchAI.depth_search()
        elif self.v.get()==1:
            action_list = SearchAI.breadth_search()
        elif self.v.get()==2:
            action_list = SearchAI.iterative_deep_search()
        self.info["text"] = "Solved: depth of solution is " + str(len(action_list)) 
        self.buttongrid.set_solution(action_list)
    
    def solve(self, board):
        
        if not self.solver.issolvable(board):
            self.info["text"] = "Not solvable"
            self.buttongrid.set_solution(None)
            return
        
        self.info["text"] = ""
        self.buttongrid.set_solution(self.solver.solve(board))
    

def main():
    
    numRows = 5
    numCols = 5
    if len(sys.argv) > 2:
        numRows = int(sys.argv[1])
        numCols = int(sys.argv[2])
    else:
        print("""Use

    {} <board_size>

to indicate the board size (default=5x5)""".format(sys.argv[0]))
    
    root = tk.Tk()
    app = App(root, numRows,numCols)
    app.master.title("Lights Out solver")
    root.mainloop()



if __name__ == '__main__':
    main()
