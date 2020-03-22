import sys
import os
import numpy as np
sys.path.append('..')

from common import pickler

data_pkl  = pickler.load('data_date')
vocab_pkl = pickler.load('vocab_date')
vocab = set('_')

def _append_vocab(sentence):
    global vocab
    vocab |= set(sentence)

def _mapping_vocab():
    char_to_id, id_to_char = {}, {}
    for i, char in enumerate(sorted(vocab)):
        char_to_id[char] = i
        id_to_char[i] = char

    return char_to_id, id_to_char

def _save_data():
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/date.txt'
    if not os.path.exists(file_path):
        print('No file.')
        return None
    q, a = [], []

    for line in open(file_path, 'r'):
        index = line.find('_')
        q.append(line[:index])
        a.append(line[index:-1])
        _append_vocab(line[:index])
        _append_vocab(line[index:-1])

    char_to_id, id_to_char = _mapping_vocab()

    x = np.zeros((len(q), len(q[0])), dtype=np.int)
    t = np.zeros((len(q), len(a[0])), dtype=np.int)

    for i, sentence in enumerate(q):
        x[i] = [ char_to_id[c] for c in list(sentence) ]
    for i, sentence in enumerate(a):
        t[i] = [ char_to_id[c] for c in list(sentence) ]

    pickler.save((x, t), 'data_date')
    pickler.save((char_to_id, id_to_char), 'vocab_date')

    return x, t, char_to_id, id_to_char

def load_data(data_type='train', seed=1984):
    if not data_pkl:
        x, t, _, _ = _save_data()
    else:
        x, t = data_pkl

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
    if not vocab_pkl:
         _, _, char_to_id, id_to_char =_save_data()
    else:
        char_to_id, id_to_char = vocab_pkl

    return char_to_id, id_to_char
