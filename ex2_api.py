"""
API for ex2, implementing the skip-gram model (with negative sampling).

"""
import math
import pickle
import random
import re
from collections import Counter

import nltk
import numpy as np
from nltk import skipgrams
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
printing = False


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


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    f = open(fn, "rb")
    sg_model = pickle.load(f)
    f.close()
    print(f"Load from {fn} file") if printing else None

    return sg_model


def save_model(model, fn):
    if fn is not None:
        with open(fn, "wb") as f:
            pickle.dump(model, f)
        print(f"saved as {fn} file") if printing else None
    else:
        print(f"NOT saved as {fn} file") if printing else None


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        """
        :param sentences: list of sentences
        :param d: Dimension of the embedding
        :param neg_samples: number of negative samples to each positive sample
        :param context: the size of the context window (not counting the target word)
        :param word_count_threshold: ignore low frequency words (appearing under the threshold)
        """
        print('Initial SkipGram...') if printing else None

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
            line_split = line.split()
            self.word_count.update(line_split)
        self.word_count = dict(self.word_count)

        # ignore low frequency words and remove stopwords
        self.word_count = {k: v for k, v in self.word_count.items()
                           if v >= word_count_threshold and k not in self.stop_words}

        # size of vocabulary
        self.vocab_size = len(self.word_count)

        # create word-index mapping
        self.word_index = {w: i for i, w in enumerate(self.word_count.keys())}
        self.index_word = {i: w for w, i in self.word_index.items()}

        self.T = np.random.rand(self.d, self.vocab_size)  # embedding representation of the words as target
        self.C = np.random.rand(self.vocab_size, self.d)  # embedding representation of the words as context

        self.V = self.combine_vectors(self.T, self.C)  # the combination of T and C

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

        v1 = self.V.T[:, indx1]
        v2 = self.V.T[:, indx2]

        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  # Cosine-Similarity

        if sim is None or math.isnan(sim):
            return 0.0

        return sim

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """
        w = w.lower()
        if w not in self.word_index:
            return []

        w_index = self.word_index[w]
        word_emb = self.V[w_index, :]

        cos_sim = np.dot(self.V, word_emb) / (
                np.linalg.norm(self.V, axis=1) * np.linalg.norm(word_emb))

        max_sorted = list(np.argsort(cos_sim)[::-1][:n + 1])  # the first index is the same word

        candidates = [self.index_word[i] for i in max_sorted if i != w_index]

        return candidates

    def create_pos_samples(self):
        """
        create to each target word in every sentence list of context in the sentence
        :return: list of dicts. each dict is for sentence, key: target, value: list of context
        """

        pos_samples = []
        for sent in self.sentences:
            sent_list = [w for w in sent.split(" ") if w in self.word_index]

            pos_forwards = list(skipgrams(sent_list, 2, self.context // 2 - 1))
            pos_reverse = list(skipgrams(sent_list[::-1], 2, self.context // 2 - 1))
            pos = pos_forwards + pos_reverse
            cs = {}
            for t, c in pos:
                t_indx = self.word_index[t]
                c_indx = self.word_index[c]
                cs.setdefault(t_indx, []).append(c_indx)
            if len(cs) > 0:
                pos_samples.append(cs)

        return pos_samples

    def add_neg_samples(self, t_index, pos_indexes):
        """
        adding random samples to each target
        :param t_index:
        :param pos_indexes:
        :return: pos_neg, y
        pos_neg: list of indexs, first the positive then negetive
        y: vector of True label, first ones then zeros [1,1,1,0,0,0,0,0]
        """
        target = self.index_word[t_index]
        neg_word = random.choices(list(self.word_count.keys()), weights=list(self.word_count.values()),
                                  k=self.neg_samples * len(pos_indexes))
        neg_indexes = [self.word_index[c] for c in neg_word if c != target]

        # transform samples to vector
        pos_y = np.ones(len(pos_indexes), dtype=int)
        neg_y = np.zeros(len(neg_indexes), dtype=int)
        y = np.concatenate((pos_y, neg_y)).reshape((-1, 1))
        pos_neg = pos_indexes + neg_indexes

        return pos_neg, y

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None, keep_train=False):
        """Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
            keep_train: if to keep training and not initial T & C from random
        """

        vocab_size = self.vocab_size
        T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

        if keep_train:
            T = self.T
            C = self.C

        # tips:
        # 1. have a flag that allows printing to standard output so you can follow timing, loss change etc.
        # 2. print progress indicators every N (hundreds? thousands? an epoch?) samples
        # 3. save a temp model after every epoch
        # 4.1 before you start - have the training examples ready - both positive and negative samples
        # 4.2. it is recommended to train on word indices and not the strings themselves.

        pos_samples = self.create_pos_samples()

        print('Start Learning...') if printing else None

        running_loss = []
        not_improved = 0
        last_loss = math.inf
        for i in range(1, epochs + 1):
            epoch_loss = []
            random.shuffle(pos_samples)
            for j, sent_dict in enumerate(pos_samples):
                # print(j, '/', len(pos_samples), 'sentence') if printing and j % 500 == 0 else None

                for t_index, pos_indexes in sent_dict.items():
                    c_indexes, y_true = self.add_neg_samples(t_index, pos_indexes)

                    hidden = T[:, t_index][:, None]
                    C_context = C[c_indexes]
                    m = len(c_indexes)

                    output_layer = np.dot(C_context, hidden)
                    y_pred = sigmoid(output_layer)
                    e = y_pred - y_true

                    loss = -(np.dot(np.log(y_pred.reshape(-1)), y_true.reshape(-1).T) + np.dot(
                        np.log(1 - y_pred.reshape(-1)),
                        (1 - y_true.reshape(-1)).T))

                    epoch_loss.append(loss / m)
                    c_grad = np.dot(hidden, e.T).T
                    t_grad = np.dot(e.T, C_context).T / m
                    # C[c_indexes, :] -= step_size * c_grad
                    np.subtract.at(C, c_indexes, step_size * c_grad)
                    T[:, [t_index]] -= step_size * t_grad

            mean_epoch_loss = np.mean(epoch_loss)
            running_loss.append(mean_epoch_loss)
            print(f'Epoch {i} Loss: ', np.round(mean_epoch_loss, 4)) if printing else None

            # check early_stopping
            if last_loss < running_loss[-1]:
                not_improved += 1
                if not_improved >= early_stopping:
                    print(f'Early Stopping: {early_stopping} Epochs not Improved') if printing else None
                    break
            else:
                not_improved = 0
            last_loss = running_loss[-1]

            # backup the last trained model (the last epoch)
            self.T = T
            self.C = C

            step_size *= 1 / (1 + step_size * i)

        print("Done Training") if printing else None

        self.T = T
        self.C = C

        save_model(self, model_path)

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
        V = np.array([])

        if combo == 0:
            V = T.T
        elif combo == 1:
            V = C
        elif combo == 2:
            V = np.multiply(C, T.T)
        elif combo == 3:
            V = np.add(C, T.T)
        elif combo == 4:
            V = np.concatenate((C, T.T), axis=1)

        self.V = V

        save_model(self, model_path)

        return V

    def find_analogy(self, w1, w2, w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """
        w1, w2, w3 = w1.lower(), w2.lower(), w3.lower()

        if w1 not in self.word_index or \
                w2 not in self.word_index or \
                w3 not in self.word_index:
            return ""

        w1_index = self.word_index[w1]
        w2_index = self.word_index[w2]
        w3_index = self.word_index[w3]

        w1_emb = self.V[w1_index, :]
        w2_emb = self.V[w2_index, :]
        w3_emb = self.V[w3_index, :]

        target_emb = w1_emb - w2_emb + w3_emb

        cos_sim = np.dot(self.V, target_emb) / (
                np.linalg.norm(self.V, axis=1) * np.linalg.norm(target_emb))
        max_sorted = list(np.argsort(cos_sim)[::-1][:5])

        # take the first word that is not w1,w2,w3
        target_index = 0
        for candidate in max_sorted:
            if candidate in [w1_index, w2_index, w3_index]:
                continue
            else:
                target_index = candidate
                break

        return self.index_word[target_index]

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

        target_w = self.find_analogy(w1, w2, w3)
        n_closest_words = self.get_closest_words(target_w, n=n - 1)

        return w4 in n_closest_words or target_w == w4
