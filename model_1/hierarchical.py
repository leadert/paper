from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath
from utils import Parameters
import pandas as pd

data_path = datapath("D:/PyCharm/PyCharm_Project/paper/data/type_relation.tsv")
type_embedding_path = "data/type_embedding"
model = PoincareModel(train_data=PoincareRelations(data_path, encoding="gbk"), size=Parameters.type_embedding_dim, negative=3)
model.train(epochs=50, print_every=5)
print(model.kv.word_vec("川菜"))
model.save(type_embedding_path)

# poincareModel = PoincareModel.load("data/type_embedding")
# print(poincareModel.kv.word_vec('东北菜'))