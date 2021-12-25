
import sudoku
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv
from image import imageio
from helper import constants
import numpy as np


def test_ai(file):
    inpt = imageio.read_image(f"test_cases/{file}.jpg")
            
    s = sudoku.Sudoku(inpt)

test_ai("example_easy")