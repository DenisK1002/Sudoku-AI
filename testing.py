
import sudoku
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv
from image import imageio
from helper import constants
import numpy as np


def imgio():
    inpt = imageio.read_image("test_cases/example_easy.jpg")
    print(inpt)
    for x in inpt:
        if len(x) != 9:
            print("wrong")
            exit(0)
            
    s = sudoku.Sudoku(inpt)
    slvd = s.solve()
    # imageio.save(inpt, slvd, "solved.jpg")


test = np.array(constants.TEST_SUDOKU)

imgio()

# for t, p in zip(test.flatten(), predict):
#     print(f"test: {t}  |   predict: {p}")