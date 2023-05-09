from ex2_api import *

fn = 'Corpus/drSeuss.txt'
norm_text = normalize_text(fn)
skipgram = SkipGram(norm_text)

# samples = skipgram.create_samples()
skipgram.learn_embeddings(epochs=50, model_path='models/model.pkl')
