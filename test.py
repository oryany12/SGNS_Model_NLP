from ex2_api import *
from nltk import skipgrams

fn = 'Corpus/drSeuss.txt'
# fn = 'Corpus/big.txt'

norm_text = normalize_text(fn)

skipgram = SkipGram(norm_text)
skipgram.learn_embeddings(epochs=100, early_stopping=10, model_path='models/model.pkl')
