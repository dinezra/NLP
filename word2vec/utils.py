import pickle
import re

import numpy as np


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    sentences = []

    # Read the text file
    with open(fn, 'r') as file:
        text = file.read()

    # Normalize the text
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = re.sub('\s+', ' ', text)
    # Remove specific punctuation marks
    text = re.sub(r'["“”.!?,]+', "", text)
    # Split into sentences
    sentences = re.split('[.!?]', text)
    # Remove leading/trailing whitespaces from each sentence
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

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
