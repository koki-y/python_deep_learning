import os.path
import pickle

dataset_dir = os.path.dirname(__file__) + "/../dataset"

def save(obj, file_name):
    with open(dataset_dir + f'/{file_name}.pkl', 'wb') as f:
        pickle.dump(obj, f, -1)

def load(file_name):
    with open(dataset_dir + f'/{file_name}.pkl', 'rb') as f:
        return pickle.load(f)

