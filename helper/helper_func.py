"""
Module to assist main python files to outsource functions.
Helper Functions
"""

from time import perf_counter

def to_array(value):
    """
    Return array of value, regardless if only 1 value
    """
    x = []
    try:
        x = [y for y in value]
    except TypeError:
        x = [value]
    
    return x
    
def sudoku_squares(board=None):
    """
    Separtating sudoku into the 9 squares
    """
    squares = []
    for sq_row in range(0, 9, 3):
        for sq_column in range(0, 9, 3):
            
            sq = []
            for i in range(sq_row, sq_row+3):
                for j in range(sq_column, sq_column+3):
                    sq += [board[i][j]]
                    
            squares += [sq]
            
    return squares


# decorator to track runtime of function
def performance(func):
    def wrapper(*args):
        print(f"Performing {func.__name__}...")
        
        start = perf_counter()
        value = func(*args)
        end = perf_counter()

        print(f"The process took {round(end - start, 4)} seconds")
        return value
    
    return wrapper