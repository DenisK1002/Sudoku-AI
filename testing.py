
import sudoku
from image import imageio


def test_ai(file):
    file = f"test_cases/{file}.jpg"
    
    inpt = imageio.read_image(file)
    
    s = sudoku.Sudoku(inpt)
    s = s.solve()
    
    image = sudoku.Sudoku.draw(file, s)
    image.show()
    
test_ai("example_easy")
