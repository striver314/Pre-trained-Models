from gensim.models import Word2Vec

# dataset
raw_sentences = ["the quick brown fox jumps over the lazy dogs", "yoyoyo you go home now to sleep"]

# word segmentation
sentences = [s.split() for s in raw_sentences]

# construct model
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

dogs_vector = model.wv['dogs']
print(dogs_vector)