import os
import numpy as np

id_to_char = {
            0 : '0', 
            1 : '1', 
            2 : '2', 
            3 : '3', 
            4 : '4', 
            5 : '5', 
            6 : '6', 
            7 : '7', 
            8 : '8', 
            9 : '9', 
            10 : '+', 
            11 : ' ',
            12 : '_'
            }

char_to_id = {
            '0' : 0, 
            '1' : 1, 
            '2' : 2, 
            '3' : 3, 
            '4' : 4, 
            '5' : 5, 
            '6' : 6, 
            '7' : 7, 
            '8' : 8, 
            '9' : 9, 
            '+' : 10,
            ' ' : 11,
            '_' : 12
            }

def load_data(data_type='train', seed=1984):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/addition.txt'

    if not os.path.exists(file_path):
        print('No file.')
        return None

    q, a = [], []

    for line in open(file_path, 'r'):
        index = line.find('_')
        q.append(line[:index])
        a.append(line[index:-1])

    x = np.zeros((len(q), len(q[0])), dtype=np.int)
    t = np.zeros((len(q), len(a[0])), dtype=np.int)

    for i, sentence in enumerate(q):
        x[i] = [ char_to_id[c] for c in list(sentence) ]
    for i, sentence in enumerate(a):
        t[i] = [ char_to_id[c] for c in list(sentence) ]

    shuffled = np.arange(len(x))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(shuffled)
    x = x[shuffled]
    t = t[shuffled]

    split_at = len(x) - len(x) // 10
    if data_type == 'test':
        return x[split_at:], t[split_at:]
    elif data_type == 'train':
        return x[:split_at], t[:split_at]

def get_vocab():
    return char_to_id, id_to_char


