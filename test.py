from ex2_api import *
from nltk import skipgrams

fn = 'Corpus/drSeuss.txt'
# fn = 'Corpus/big.txt'
#
norm_text = normalize_text(fn)
skipgram = SkipGram(norm_text)

# samples = skipgram.create_samples()
skipgram.learn_embeddings(epochs=1000, model_path='models/model.pkl')
print(skipgram.get_closest_words('king'))


# sent = "Insurgents killed in ongoing fighting frank ory".split()
#
# print(list(skipgrams(sent, 2, 2//2-1)))
# print(list(skipgrams(sent[::-1], 2, 2//2-1)))
