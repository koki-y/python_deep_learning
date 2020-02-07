from PIL import Image
import numpy as np

def show_image(img):
    Image.fromarray(np.uint8(img.reshape(28, 28))).show()

def show_answer(ans):
    ans = list(ans)
    print("The answer is " + str(ans.index(max(ans))) + ".")

