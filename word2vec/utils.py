import pickle
import re

import numpy as np


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    sentences = []

    file = open(fn, "r", encoding="cp1252")
    lines = file.readlines()
    file.close()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        line = re.sub(r'["|“|”|,|!|?|.]+', "", line)
        line = line.lower()
        try:
            sentences.append(line.split())
        except:
            continue
    return sentences
def sigmoid(x): return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """
    with open(fn, 'rb') as file:
        sg_model = pickle.load(file)

    return sg_model

def _load_model(model_path):
    """Loads a trained word2vec model from the specified file.

    Args:
        model_path: The path to the model file.

    Returns:
        The loaded word2vec model.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model