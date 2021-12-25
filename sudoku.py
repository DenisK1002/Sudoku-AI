from copy import deepcopy
from sys import exc_info
from helper.helper_func import to_array, sudoku_squares, performance
from functools import lru_cache

from image import imageio


class Sudoku():
    """
    Sudoku Class to represents Sudoku game object.\n\n
    It is able to solve the sudoku and output the finished game as 'output.jpg'
    """
    def __init__(self, sudoku) -> None:
        """
        Class requires sudoku game input as
        following structure: \n
        sudoku -> [[row_of_int], [2nd_row_of_int], ... ]
        replace empty cells with None
        """
        
        # check if sudoku is invalid
        if not self.valid(sudoku):
            raise ValueError
        
        self.sudoku = sudoku
        self.i = len(sudoku[0])
        self.j = len(sudoku)
        self.domains = self.retrieve_domains()
        

    def retrieve_domains(self):
        """
        Returns initial domain for each cell in sudoku
        """
        d = dict()
        for i in range(self.i):
            for j in range(self.j):
                d[i,j] = [i for i in range(1, 10)]

        return d


    def cross(self, x):
        """
        Returns innnersquare, vertical and horizontal variables corresponding to v
        """

        xi = x[0]
        xj = x[1]
        crossing = []

        # horizontal neighbors
        for i in range(self.i):
            if i != xi:
                crossing.append((i, xj))

        # vertical neighbors
        for j in range(self.j):
            if j != xj:
                crossing.append((xi, j))

        def square(i_start, i_end, j_start, j_end):
            sq = []
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    sq += [(i, j)]
            return sq

        def insert(sq):
            for c in sq:
                if c != x and c not in crossing:
                    crossing.append(c)

        matrizes = [
            square(0, 3, 0, 3),
            square(0, 3, 3, 6),
            square(0, 3, 6, 9),

            square(3, 6, 0, 3),
            square(3, 6, 3, 6),
            square(3, 6, 6, 9),

            square(6, 9, 0, 3),
            square(6, 9, 3, 6),
            square(6, 9, 6, 9)
        ]

        # innersquare crossings
        for m in matrizes:
            if x in m:
                insert(m)

        return set(crossing)
    
    
    def valid(self, board):
        """
        Checks if input board is valid
        """
        
        len_y = len(board)
        len_x = all([lambda x: len(x) == 9 for x in board])
        
        if len_y == 9 and len_x:
            return True
        
        return False

    
    def prefill_assignment(self, assignment=dict()):
        """
        If Sudoku happens to have prefilled cells, add them to the assignment 
        """
        for i in range(self.i):
            for j in range(self.j):
                if self.sudoku[i][j] != None:
                    assignment[i,j] = self.sudoku[i][j]

        return assignment


    def preset_domains(self):
        """
        If Sudoku happens to have prefilled cells,  \n
        set domain of that cell to existing value
        """
        for i, j in self.domains:
            prefill_number = self.sudoku[i][j]
            if prefill_number != None:
                self.domains[i, j] = prefill_number


    def revise(self, x1, x2):
        """
        Make Variable x1 arc-consistent with variable x2 \n
        by checking if a possible value for x2 exists by choosing x1. \n
        \n
        Returns True if revision (change) was made, else False
        """
        # loop variable
        change = False

        for a in to_array(self.domains[x1]):
            # keep track if option exists
            option = False

            # considering each domain of x2
            for b in to_array(self.domains[x2]):
                # if b is not a (which is a constraint of a sudoku)
                # if so there is an option, else false
                if b != a:
                    option = True

            # if there is no option, remove a from x1's domain
            if not option:
                try:
                    self.domains[x1].remove(a)
                except AttributeError: # domains only have 1 value left (int)
                    self.domains[x1] = to_array(self.domains[x1])
                    self.domains[x1].remove(a)
                    
                change = True
                
        return change

    def ac3(self, arcs=None):
        """
        Make assignment consistent to binary constraints
        Binary Constraints in the sudoku csp are:
            - uniqueness off number in:
                - horizontal and
                - vertical line
                - inner-cell (e.g. 3x3)
        """
        if arcs == None:
            arcs = []
            for x in self.domains:
                for c in self.cross(x):
                    arcs.append((x, c))

        while len(arcs) != 0:
            x1, x2 = arcs.pop(0)
            if self.revise(x1, x2):
                if len(self.domains[x1]) == 0:
                    return False

                for y in [list(self.cross(x1)).remove(x2)]:
                    if y != None:
                        arcs += [(y, x1)]
                        
        return True


    def assignment_board(self, assignment):
        """
        Takes assignment, returns starting board including assignment variables
        """
        
        assignment_board = deepcopy(self.sudoku)
        for i, j in assignment:
            assignment_board[i][j] = assignment[i, j]
            
        return assignment_board
    
    
    def complete(self, assignment):
        """
        Returns True if sudoku is solved and correct. False otherwise.
        """
        
        assignment_board = self.assignment_board(assignment)

        for row in assignment_board:
            for value in row:
                
                if value not in range(1, 10):
                    return False
                
                if value == None:
                    return False
        
        if self.conflicts(assignment):
            return False
        
        return True

        
    def select_unassigned_variable(self, assignment):
        """
        Returns variable which has no domain set in assignment.\n
        Ordered by least amount of domain values.
        """
        
        try:
            return [v for v in self.domains if v not in assignment][0]
        except IndexError: # return last assigned
            return list(assignment.keys())[-1]
            
            
        minimum = {
            v: len(to_array(self.domains[v]))
            for v in self.domains if v not in assignment
        }
        
        # if no variable remaining, return the last variable that got assigned
        if len(minimum) == 0:
            return list(assignment.keys())[-1]
        
        # sort by length of domains
        minimum_sorted = sorted(minimum, key=lambda v: minimum[v])
        
        # only lowest n of domains
        for m in minimum_sorted:
            if minimum[minimum_sorted[0]] != minimum[m]:
                minimum_sorted.remove(m)

        # degree
        if len(minimum_sorted) > 1:
            
            degree = []
            for v in minimum_sorted:
                neighbours = [k for k in self.cross(v) if k not in assignment]
                degree.append([v, len(neighbours)])

            # return first variable which has least amount of domains and highest amount of neighbours
            return sorted(degree, key=lambda d: d[1], reverse=True)[0][0]
        
        # return first element (least domains)
        return minimum_sorted[0]


    def domain(self, var):
        """
        Returns sorted domain of var. \n
        Sorts accordingly to the number of excluded options of crossings (neighbours -> crossing() )
        """

        crossings = {
            c: to_array(self.domains[c])
            for c in self.cross(var)
        }
        
        excluded = {
            d: 0
            for d in self.domains[var]
        }
        for d in excluded:
            for c_d in crossings.values():
                if d in c_d:
                    excluded[d] += 1

        return sorted(excluded, key=lambda x: excluded[x])
        

    def conflicts(self, assignment):
        """
        Returns True if Sudoku has conflicts. False Otherwise.
        Conflicts are:
            if value appears multiple times in:
                - horizontal row
                - vertical row or
                - in the 3x3 square
        """
        
        assignment_board = self.assignment_board(assignment)

        # horizontal conflicts
        for row in assignment_board:
            for value in row:
                if value != None and row.count(value) > 1:
                    return True
        
        # vertical conflicts
        for column in range(self.i):
            column_values = []
            for row in range(self.j):
                column_values.append(assignment_board[row][column])
            
            for cv in column_values:
                if cv != None and column_values.count(cv) > 1:
                    return True
        
        # innersquare conflicts
        squares = sudoku_squares(assignment_board)
        for sq in squares:
            for value in sq:
                if value != None and sq.count(value) > 1:
                    return True

        return False


    def backtrack(self, assignment):
        """
        Searches through all possible board states and backtracks if conflict is found
        """

        # sudoku complete -> return sudoku as array
        if self.complete(assignment):
            return self.assignment_board(assignment)

        # select cell yet unassigned
        variable = self.select_unassigned_variable(assignment)

        # consider each remaining domain value
        for value in self.domain(variable):

            # if the assignment is still without conflicts
            if not self.conflicts(assignment):
                
                # try out value for variable
                assignment[variable] = value

                # result = next state (until either complete or None)
                result = self.backtrack(assignment)

                # if end != None (if it is complete)
                if result != None:
                    return result
                
                # else (if it has conflicts) delete variable out of assignment
                assignment.pop(variable)

        return None


    @performance
    def solve(self):
        """
        Solve Sudoku Instance.\n
        Returns None if no solution.
        """
        assignment = self.prefill_assignment()
        self.preset_domains()
        self.ac3()
        return self.backtrack(assignment)
    
    
    def draw(self, assignment_board, path):
        """
        Generate an Image of the solved sudoku.\n
        Saves to given path
        """
        
        image = imageio.draw_solved()
        
        raise NotImplementedError
    
    