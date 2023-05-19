import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

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
        self.V = None
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
        try: nltk.corpus.stopwords.words("english")
        except: nltk.download("stopwords", quiet=True)

        stop_word = set(stopwords.words("english"))
        words_conut = collections.Counter()
        for sentence in sentences:
            words_conut.update(sentence)
        word_count_non_stop = {}
        for word, count in words_conut.items():
            if word not in stop_word:
                word_count_non_stop[word] = count
        return word_count_non_stop

    def _word_index(self):
        """

        :return:
        """
        word_index = {}
        for index,word in enumerate(self.word_counts.keys()):
            word_index[word]=index
        return word_index
    def get_emb(self,w):
        return self.V[self.word_index[w]]

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
            mone=np.dot(emb1, emb2)
            # Compute the cosine similarity
            if np.all(emb1 == 0):
                return 0
            sim = mone/ (np.linalg.norm(emb1) * np.linalg.norm(emb2))

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
        hidden_layer = self.V[input_layer_id][:, None]

        output_layer = np.dot(hidden_layer, np.array(self.C ))
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
        model_path = '../models'

        print('=' * 40 + " Start Preprocessing " + '=' * 41)
        vocab_size = len(self.word_index)
        T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

        # Prepare training examples (positive and negative samples)
        training_data = self.prepare_training_examples()
        print('=' * 40 + " Finish Preprocessing " + '=' * 40)

        best_loss = float('inf')
        patience = early_stopping
        print('=' * 40 + " Start Training " + '=' * 46)

        for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
            total_loss = 0.0

            # Training loop
            for key, val in training_data:
                # Input layer x T = Hidden layer
                input_layer_id = self.word_index[key]
                input_layer = np.zeros(self.vocab_size, dtype=int)
                input_layer[input_layer_id] = 1
                input_layer = np.vstack(input_layer)

                hidden = T[:, input_layer_id][:, None]

                output_layer = np.dot(C, hidden)
                y = sigmoid(output_layer)

                # calculate gradient
                e = y - val.reshape(self.vocab_size, 1)
                outer_grad = np.dot(hidden, e.T).T
                inner_grad = np.dot(input_layer, np.dot(C.T, e).T).T
                C -= step_size * outer_grad
                T -= step_size * inner_grad

                # calculate loss
                loss = np.sum(e ** 2)
                total_loss += loss

            avg_loss = total_loss / len(training_data)
            tqdm.write(f" Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = early_stopping
            else:
                patience -= 1
                if patience == 0:
                    print(f"Training stopped early. Loss did not improve for {early_stopping} epochs.")
                    break

            # Backup the last trained model (the last epoch)
            self.T = T
            self.C = C
            # with open(os.path.join(model_path, f"temp_model_epoch_{epoch + 1}.pickle"), "wb") as f:
            #     pickle.dump(self, f)

            step_size *= 1 / (1 + step_size * epoch)

        print("\nTraining completed.")

        self.T = T
        self.C = C

        with open(os.path.join(model_path, 'final_model.pickle'), "wb") as f:
            pickle.dump(self, f)

        print(f"\nModel save to {model_path} as final_model.pickle\n")
        self.V = T
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

        if combo == 0:
            V = T
        elif combo == 1:
            V = C
        elif combo == 2:
            V = (T.T + C) / 2
        elif combo == 3:
            V = T.T + C
        elif combo == 4:
            V = np.concatenate((T.T,C), axis=1)
        else:
            raise ValueError("Invalid combo value. Must be 0, 1, 2, 3, or 4.")

            # Save the combined embedding matrix
        if model_path is not None:
            with open(model_path, "wb") as f:
                pickle.dump(V, f)
                print(f"Vectors save to {model_path} \n")

        else:
            # Handle the case when model_path is None
            print("Warning: model_path is not provided. The combined embeddings will not be saved.")
        self.V = V
        return V

    def find_analogy(self, w1, w2, w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """

        if w1 not in self.word_index or w2 not in self.word_index or w3 not in self.word_index:
            return []  # One or more words are not in the vocabulary

        # Get the word indices from the word_index
        w1_idx = self.word_index[w1]
        w2_idx = self.word_index[w2]
        w3_idx = self.word_index[w3]

        # Get the corresponding word embeddings from the embedding matrix
        w1_embedding = self.V[w1_idx]
        w2_embedding = self.V[w2_idx]
        w3_embedding = self.V[w3_idx]

        v_diff = w2_embedding - w1_embedding
        analogy_embedding = v_diff + w3_embedding

        # Find the closest word to the analogy embedding
        closest_word_idx = np.argmax(np.dot(self.V, analogy_embedding))

        # Retrieve the word from the word_index
        w = list(self.word_index.keys())[list(self.word_index.values()).index(closest_word_idx)]
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

        if w1 not in self.word_index or w2 not in self.word_index or w3 not in self.word_index or w4 not in self.word_index:
            return False

        # Calculate the vector representation of the analogy
        v1 = self.V[self.word_index[w1]]
        v2 = self.V[:, self.word_index[w2]]
        v3 = self.V[ self.word_index[w3]]
        analogy_vector = v1 - v2 + v3

        # Get the n closest words to the analogy vector
        closest_words = self.get_closest_words(analogy_vector, n=n)

        return w4 in closest_words

