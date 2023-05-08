from ex2_api import *

fn = 'Corpus/drSeuss.txt'
norm_text = normalize_text(fn)
print(norm_text)
skipgram = SkipGram(norm_text)

samples = skipgram.create_samples()
