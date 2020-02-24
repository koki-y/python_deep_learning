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

def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('query not found.')
        return

    print(f'Query: {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word[i]} : {similarity[i]}')

        count += 1
        if count >= top:
            return

