import random
import os

from AUTOGRAD import Utils
## This file loads the data and returns the feature and target values.

def load_and_prep_data(filepath, shuffle_initial=False):
    """
    Ingests data from a CSV, optionally performs an initial shuffle,
    and returns the features (xs) and targets (ys).
    """
    if not filepath:
        print("[Data Engine]: No filepath provided. Using default OR Gate fallback.")
        xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        ys = [0.0, 1.0, 1.0, 1.0]
        
    elif  not os.path.isfile(filepath):
        raise FileNotFoundError(f"[Data Engine]: NO dataset exists at '{filepath}'")

    else:
        print(f"[Data Engine]: Ingesting dataset from {filepath}...")
        xs, ys = Utils.load_csv(filepath)

    if shuffle_initial:
        print("[Data Engine]: Performing initial dataset shuffle...")
        combined = list(zip(xs, ys))
        random.shuffle(combined)
        
        # unzip back into separate lists
        xs, ys = zip(*combined)
        xs, ys = list(xs), list(ys)

    return xs, ys