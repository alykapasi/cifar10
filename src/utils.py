import pickle

def unpickle(file: str) -> dict:
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='bytes')