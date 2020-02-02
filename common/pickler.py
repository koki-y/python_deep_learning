import os.path
import pickle

dataset_dir = os.path.dirname(__file__) + "/../dataset"
save_file = dataset_dir + "/two_layer_net.pkl"

def save(params):
    with open(save_file, 'wb') as f:
        pickle.dump(params, f, -1)

def load():
    with open(save_file, 'rb') as f:
        return pickle.load(f)

