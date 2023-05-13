import pickle
import pandas as pd
import numpy as np
import os,time, re, sys, random, math, collections, nltk

from nltk import skipgrams
from nltk.corpus import stopwords

from word2vec.utils import *


#static functions
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Din Ezra', 'id': '206065989', 'email': 'ezradin@post.bgu.ac.il'}


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context #the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold #ignore low frequency words (appearing under the threshold)
        self.T = []  # embedding matrix
        self.C = []  # embedding matrix
        self.word_counts = self._word_count(sentences)
        self.word_index = self._word_index()
        self.vocab_size = len(self.word_counts)

    def _word_count(self,sentences):
        """

        :param sentences:
        :return:
        """
        if not nltk.corpus.stopwords.words("english"):
            nltk.download("stopwords", quiet=True)

        stop_word = set(stopwords.words("english"))
        words_conut = collections.Counter()
        for sentence in sentences:
            for word in sentence.split():
                if word not in stop_word: words_conut.update(word)
        return dict(words_conut)

    def _word_index(self):
        """

        :return:
        """
        word_index = {}
        for index,word in enumerate(self.word_counts.keys()):
            word_index[word]=index
        return word_index
    def get_emb(self,w):
        return self.T[:, self.word_index[w]]

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        sim  = 0.0 # default
        # Check if both words are present in the Word2Vec model's vocabulary
        if w1 in self.word_counts and w2 in self.word_counts:
            # Get the word embeddings for w1 and w2
            emb1 = self.get_emb(w1)
            emb2 = self.get_emb(w2)

            # Compute the cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return sim # default

    def get_closest_words(self, word, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            word: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """

        if word not in self.word_index:
            return []  # default

        output_layer = self.feed_forward(word)
        n = min(n, self.vocab_size)

        candidates = []
        for candidate_word, index in self.word_index.items():
            candidates.append((candidate_word, output_layer[index]))

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        closest_words = [word for word, score in candidates]
        return closest_words[:n]

    def feed_forward(self, word):
        """Returns a normalized output layer for a word

        Args:
            word: word to get output for
        """

        input_layer_id = self.word_index[word]
        hidden_layer = self.T[:, input_layer_id][:, None]

        output_layer = np.dot(self.C, hidden_layer)
        normalized_output = sigmoid(output_layer)

        return normalized_output

    def prepare_training_examples(self):
        """Prepares the training examples for the word2vec model.

        Returns:
            A list of tuples (target, context, label) representing the training examples.
        """

        # create learning vectors
        training_examples  = []
        for sentence in self.sentences:
            dic = {}

            # create positive and negative lists
            positive_lst = list(skipgrams(sentence.split(), int(self.context / 2), 1))
            positive_lst += [(tup[1], tup[0]) for tup in positive_lst]
            negative_lst = []
            for _ in range(self.neg_samples):
                negative_lst += [
                    (word, random.choice(list(self.word_counts.keys())))
                    for word in sentence.split()
                ]

            pos = {}
            for x, y in positive_lst:
                if x not in self.word_counts or y not in self.word_counts:
                    continue
                pos.setdefault(x, []).append(y)
            neg = {}
            for x, y in negative_lst:
                if x not in self.word_counts or y not in self.word_counts:
                    continue
                neg.setdefault(x, []).append(y)

            # create the learning context vector
            for key, val in pos.items():
                context_vector = np.zeros(self.vocab_size, dtype=int)
                for v in val:
                    context_vector[self.word_index[v]] += 1
                for v in neg[key]:
                    context_vector[self.word_index[v]] -= 1
                dic[key] = context_vector

            training_examples += dic.items()
        return  training_examples

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
        """Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """
        print('='*40+"Start Preprocessing"+'=')
        vocab_size = len(self.word_index)
        T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

        # Prepare training examples (positive and negative samples)
        training_data = self.prepare_training_examples()
        print('='*40+"Finish Preprocessing"+'=')

        best_loss = float('inf')
        patience = early_stopping
        print('='*40+"Start Training"+'=')

        for epoch in range(epochs):
            total_loss = 0.0

            # Training loop
            for key, val  in training_data:
                # Input layer x T = Hidden layer
                input_layer_id = self.word_index[key]
                input_layer = np.zeros(self.vocab_size, dtype=int)
                input_layer[input_layer_id] = 1
                input_layer = np.vstack(input_layer)

                hidden = T[:, input_layer_id][:, None]

                # Hidden layer x C = Output layer
                output_layer = np.dot(C, hidden)
                y = sigmoid(output_layer)

                # calculate gradient
                e = y - val.reshape(self.vocab_size, 1)
                outer_grad = np.dot(hidden, e.T).T
                inner_grad = np.dot(input_layer, np.dot(C.T, e).T).T
                C -= step_size * outer_grad
                T -= step_size * inner_grad

                # backup the last trained model (the last epoch)
            self.T = T
            self.C = C
            with open("temp.pickle", "wb") as f:
                pickle.dump(self, f)

            step_size *= 1 / (1 + step_size * i)
        print("done training")

        self.T = T
        self.C = C

        with open(model_path, "wb") as f:
            pickle.dump(self, f)

        return T, C


    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """

        # TODO

        return V

    def find_analogy(self, w1,w2,w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """

        #TODO

        return w

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """

        # TODO

        return False