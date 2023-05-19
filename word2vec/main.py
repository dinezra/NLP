from word2vec_class import *
from word2vec_class import SkipGram
from utils import *

#PARAMETERS
combo_num = 0
# path_data = '../Data/big.txt'
path_data = '../Data/drSeuss.txt'
model_path = './models'

vec_path = os.path.join(model_path,f"combine_{combo_num}_vectors.pickle")

sentences = normalize_text(path_data)

model = SkipGram(sentences)
model.learn_embeddings(model_path=model_path,
                       epochs=100)
