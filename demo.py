from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def show_image(img):
    Image.fromarray(np.uint8(img.reshape(28, 28))).show()

def show_answer(ans):
    ans = list(ans)
    print("The answer is " + str(ans.index(max(ans))) + ".")

def show_spiral_data(x):
    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']
    for i in range(CLS_NUM):
        plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
    plt.show()
