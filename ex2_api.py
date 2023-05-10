"""
API for ex2, implementing the skip-gram model (with negative sampling).

"""
import pickle
import pandas as pd
import numpy as np
import os, time, re, sys, random, math, collections, nltk
from collections import Counter
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)


# static functions
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Oryan Yehezkel', 'id': '311495824', 'email': 'oryanyeh@post.bgu.ac.il'}


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    f = open(fn, "r")

    lines = f.readlines()
    f.close()

    sentences = []

    for l in lines:
        l = l.strip()

        if l == "" or l is None:
            continue

        l = re.sub(r'["|“|”|.|!|?|,]+', "", l)
        l = l.lower()

        sentences.append(l)

    return sentences


def sigmoid(x): return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    f = open(fn, "rb")
    sg_model = pickle.load(f)
    f.close()

    return sg_model


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        """

        :param sentences: list of sentences
        :param d: Dimension of the embedding
        :param neg_samples: number of negative samples to each positive sample
        :param context: the size of the context window (not counting the target word)
        :param word_count_threshold: ignore low frequency words (appearing under the threshold)
        """
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context  # the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold  # ignore low frequency words (appearing under the threshold)
        self.word_index = {}  # word-index mapping
        self.word_count = Counter()  # word:count dictionary
        self.stop_words = set(stopwords.words("english"))

        # populate word_count
        for line in sentences:
            self.word_count.update(line.split())
        self.word_count = dict(self.word_count)

        # ignore low frequency words and remove stopwords
        self.word_count = {k: v for k, v in self.word_count.items()
                           if v >= word_count_threshold and k not in self.stop_words}

        # size of vocabulary
        self.vocab_size = len(self.word_count)

        # create word-index mapping
        self.word_index = {w: i for i, w in enumerate(self.word_count.keys())}

        self.T = np.random.rand(self.d, self.vocab_size)  # embedding representation of the words as target
        self.C = np.random.rand(self.vocab_size, self.d)  # embedding representation of the words as context

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        sim = 0.0  # default
        if w1 not in self.word_index or w2 not in self.word_index:
            return sim

        indx1 = self.word_index[w1]
        indx2 = self.word_index[w2]

        v1 = self.T[indx1]
        v2 = self.T[indx2]

        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  # Cosine-Similarity

        return sim

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """

    def create_samples(self):
        total_samples = []  # list of tuple of 2 (str: target, vector: (0,1,-1)^|V| | 1 for pos, -1 for neg)
        for sent in self.sentences:
            sent_list = sent.split(" ")

            for i in range(len(sent_list)):
                target = sent_list[i]
                if target not in self.word_count: continue
                target_indx = self.word_index[target]

                pos_samples = set()
                # create positive samples
                context = sent_list[i - self.context:i] + sent_list[i + 1:i + 1 + self.context]
                for c in context:
                    if c not in self.word_count or c == target: continue
                    pos_samples.add(self.word_index[c])

                neg_samples = set()
                # create negative samples
                neg_word = random.choices(list(self.word_count.keys()), weights=list(self.word_count.values()),
                                          k=self.neg_samples * len(pos_samples))
                neg_samples.update([self.word_index[c] for c in neg_word if c != target])

                # transform samples to vector
                pos_y = np.ones(len(pos_samples), dtype=int)
                neg_y = np.zeros(len(neg_samples - pos_samples), dtype=int)
                y = np.concatenate((pos_y, neg_y)).reshape((-1, 1))
                all_samples = list(pos_samples) + list(neg_samples - pos_samples)

                if len(all_samples) > 0:
                    total_samples.append((target_indx, all_samples, y))
        return total_samples

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None, printing=True):
        """Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """

        vocab_size = self.vocab_size
        T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

        # tips:
        # 1. have a flag that allows printing to standard output so you can follow timing, loss change etc.
        # 2. print progress indicators every N (hundreds? thousands? an epoch?) samples
        # 3. save a temp model after every epoch
        # 4.1 before you start - have the training examples ready - both positive and negative samples
        # 4.2. it is recommended to train on word indices and not the strings themselves.

        # TODO
        samples = self.create_samples()
        running_loss = []
        for i in range(1, epochs + 1):
            epoch_loss = []
            for target_index, c_indexs, y_true in samples:
                input_layer = np.zeros(self.vocab_size, dtype=int)
                input_layer[target_index] = 1
                input_layer = np.vstack(input_layer)

                hidden = T[:, target_index][:, None]
                C_context = C[c_indexs]
                m = len(c_indexs)

                output_layer = np.dot(C_context, hidden)
                y_pred = sigmoid(output_layer)
                e = y_pred - y_true
                loss = -(np.dot(np.log(y_pred.reshape(-1)), y_true.reshape(-1).T) + np.dot(
                    np.log(1 - y_pred.reshape(-1)),
                    (1 - y_true.reshape(-1)).T))
                """
                I HAVE PROBLEM WITH THE Y OF SAMPLES - HOW TO REPRESENT THE Y_TRUE VECTOR(VAL PARAM)
                I HAVE PROBLEM WITH THE LOSS - AFTER FEW ITERATION THE LOSS IS NEGATIVE!
                """

                epoch_loss.append(loss)
                c_grad = np.dot(hidden, e.T).T
                # t_grad = np.dot(input_layer, np.dot(C_context.T, e).T).T
                t_grad = np.dot(e.T, C_context).T / m
                C[c_indexs, :] -= step_size * c_grad
                T[:, [target_index]] -= step_size * t_grad

            mean_epoch_loss = np.mean(epoch_loss)
            running_loss.append(epoch_loss)
            print(f'Epoch {i} Loss: ', np.round(mean_epoch_loss, 4)) if printing else None

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
        print(f"saved as {model_path} file") if printing else None

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

    def find_analogy(self, w1, w2, w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """

        # TODO

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
