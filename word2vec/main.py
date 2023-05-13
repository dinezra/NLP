from word2vec_class import *
from word2vec_class import SkipGram
from utils import *

path_data = '../Data/big.txt'
path_data = '../Data/drSeuss.txt'
model_path = './models'
vec_path = os.path.join(model_path,"combine_vectors.pickle")
sentences = normalize_text(path_data)

model = SkipGram(sentences)
model.learn_embeddings(model_path=model_path,
                       epochs=10)

V = model.combine_vectors( model.T,
                     model.C,
                     combo=0,
                     model_path=model_path)

