from ex2_api import *
from nltk import skipgrams

fn = 'Corpus/drSeuss.txt'
# fn = 'Corpus/big.txt'

norm_text = normalize_text(fn)

skipgram = SkipGram(norm_text)
skipgram.learn_embeddings(epochs=1000,early_stopping=100, model_path='models/model.pkl')
print(skipgram.get_closest_words('cat'))


# norm_text = ['prince future king', 'daughter princess', 'son prince', 'man king', 'woman queen', 'princess queen',
#         'queen king rule realm', 'prince strong man', 'princess beautiful woman', 'royal family king queen children',
#         'prince boy', 'boy man']
# skipgram = SkipGram(norm_text, d=2, neg_samples=1, context=2, word_count_threshold=1)
