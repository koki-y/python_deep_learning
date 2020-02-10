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

def show_spiral_with_boundary(x, model):
    # Create 2D vector (1000 * 1000)
    h = 0.001
    x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
    y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]

    # Calcurate the score used for classification into three class.
    score = model.predict(X)               # Calclurate probability
    predict_cls = np.argmax(score, axis=1) # Pick the index of max probability (This is the answer class)
    Z = predict_cls.reshape(xx.shape)      # Reshape for ploting

    # Plot classification boundary
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z)  

    # Plot spiral
    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']
    for i in range(CLS_NUM):
        ax.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])

    plt.show()

