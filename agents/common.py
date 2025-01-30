import os
import pickle

import numpy as np


def load_transitions(path):
    batch_files = [f for f in os.listdir(path) if f.startswith("transitions_batch_") and f.endswith(".pkl")]
    batch_files.sort(key=lambda x: int(x.rsplit('_', 1)[-1].split('.')[0]))
    transitions = [transition for batch_file in batch_files
                   for transition in pickle.load(open(os.path.join(path, batch_file), 'rb'))]

    return np.array(transitions, dtype=object)
