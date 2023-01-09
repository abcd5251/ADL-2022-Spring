from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

(count, dimensions) = glove2word2vec("./glove.840B.300d.txt", "./gensim_glove.840B.300d.txt")
