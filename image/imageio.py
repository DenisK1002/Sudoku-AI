"""
Module handles image Input/Output

Referring to these sources: 
https://docs.opencv.org/4.5.3/d5/d0f/tutorial_py_gradients.html
"""

# turn off tf logging
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2 as cv
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import sys
from math import floor
from helper.helper_func import performance
from tensorflow.keras.models import load_model


model = load_model("image/number_recognition.h5")

# remove truncuation
np.set_printoptions(threshold=sys.maxsize)


def draw_solved(input_image, solved_board):
    """
    Draws the completed sudoku onto input image, returns array
    """
    input_image = cv.imread(input_image, 0)
    input_image = threshold_filter(input_image, 60)
    input_image = crop_image(input_image, 0)
    solved_board = np.ndarray.flatten(np.array(solved_board))
    
    # coordinates of cells
    coord_cells = coordinates(input_image)
    
    image = Image.fromarray(input_image) 
    image = image.convert("RGB")
    font = ImageFont.truetype("arial.ttf", 48)
    draw = ImageDraw.Draw(image)
    for c in coord_cells:
        draw.text(c[0], str(solved_board[c[1]]), "green", font=font) 
    
    return image

def coordinates(image):
    """
    Return coordinates of empty cells in image.
    ((tuple, cord), index)
    """
    sudoku_cells = cells(image, cell_size(image))
    
    coords = []
    for i, cell in enumerate(sudoku_cells):
        cell_number = number(cell)
        
        if cell_number == None:
            coords.append([center(image, i), i])
    
    return coords
    
def center(image, index):
    """
    Returns center coordinates of cell
    """
    
    height, width  = cell_size(image)

    w = index / 9
    
    if w >= 1:
        w = w - floor(w)
        
    w = w*10
    w = floor(w)
    
    coords = (w * width + width//2, height * (index//9) * 0.95 + height//2)
    
    return coords

@performance
def read_image(input_image_path):
    """
    Read input image and return array containing prefilled numbers
    """
    # read image
    img = cv.imread(input_image_path, 0)
    
    # extract cells
    img = threshold_filter(img, 80)

    img = crop_image(img, 0)
    sudoku_cells = cells(img, cell_size(img))
    
    # each cell's image to number converting using cnn model
    numbers = []
    for cell in sudoku_cells:
        numbers.append(number(cell))

    numbers_rows = []
    for start in range(0, len(numbers), 9):
        row = numbers[start:start + 9]
        numbers_rows.append(row)
    
    return numbers_rows
    
def number(cell):
    """
    Returns number in picture using pretrained number recognition model
    """
    
    cell = threshold_filter(cell, 170)
    cell = crop_image(cell, 20)
   
    cell = cv.resize(cell, dsize=(28,28), interpolation=cv.INTER_CUBIC)

    cell = cell.astype('float32')
    cell /= 255
    
    cell = np.expand_dims(cell, 0)
    cell = np.expand_dims(cell, 3)
    
    prediction = model.predict(cell)[0]
    
    prediction = {
        "prediction": prediction.tolist().index(max(prediction)), 
        "probability": max(prediction) }
    
    if prediction["probability"] > 0.8:
        return prediction["prediction"]
    elif prediction["probability"] > 0.4 and prediction["prediction"] == 7: # training set falsely labels ones to sevens with 40-50 % chance
        return 1
    else:
        return None


def crop_image(image, margin):
    """
    Cropping image to the bounds of sudoku
    """  
    
    non_zeros_x, non_zeros_y = np.nonzero(image)

    try:
        x_left, x_right = non_zeros_x.min(), non_zeros_x.max()
        y_left, y_right = non_zeros_y.min(), non_zeros_y.max()
    except ValueError: # raised if image is empty (cell with no value)
        return image    
    
    img = image[x_left : x_right,  y_left : y_right]

    # adding margin
    if margin > 0:
        # vertical
        zeros = np.zeros(len(img[0]))
        
        for _ in range(margin):
            img = np.concatenate([[zeros], img])
            img = np.concatenate([img, [zeros]])
            
        # horizontal
        zeros = np.zeros((len(img), 1))
        
        for _ in range(margin):
            img = np.hstack([img, zeros])
            img = np.hstack([zeros, img])

    return img


def threshold_filter(image, threshold):
    
    func = lambda i: 0 if i < threshold else i
    
    result = list([list(map(func, i)) for i in image])
            
    return np.array(result, dtype='uint8')
    
def cell_size(image):
    """
    Returns tuple of sudoku cell size
    """
    
    return (len(image) // 9, len(image[0]) // 9)


def cells(image, cell_size):
    """
    Returns numpy array of sudoku cells
    """
    
    s_cells = []
    for line in range(0, len(image), cell_size[0]):
        for column in range(0, len(image[0]), cell_size[1]):
            cell = np.array([c[column:column + cell_size[1]] for c in image[line:line+cell_size[0]]], dtype='uint8')

            if cell.size > 10000:
                s_cells.append(cell)

    return np.array(s_cells, dtype='uint8')


def to_txt(array):
    with open("output.txt", "w") as f:
        for pixel in array:
            f.write(f"{pixel}")


def save(array, path):
    """
    Saving numpy array to path as image
    """
    
    img = Image.fromarray(array)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    img.save(path)
    