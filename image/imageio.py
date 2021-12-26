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
import PIL
import sys

from tensorflow.keras.models import load_model


model = load_model("image/number_recognition.h5")

# remove truncuation
np.set_printoptions(threshold=sys.maxsize)


def draw_solved(input_image, board):
    """
    Draws the completed sudoku onto input image, returns array
    """
    # TEST CODE    
    # image = PIL.Image.open("solved.png")
    # font = PIL.ImageFont.truetype("arial.ttf", 26)
    # draw = PIL.ImageDraw.Draw(image)
    # draw.text((130,125), "0", (255,255,255), font=font)
    # image.save("output1.jpg")
    raise NotImplementedError


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
    cell = crop_image(cell, 30)
    
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
    
    img = PIL.Image.fromarray(array)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    img.save(path)
    

def center(image):
    """
    Returns center coordinates of image
    """
    raise NotImplementedError