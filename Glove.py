from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from configuration import ConfigClass

class Glove:

    def __init__(self):
        self.config=ConfigClass()
        self.local_address = r"C:\Users\benro\OneDrive\Desktop\glove.twitter.27B.25d.txt"
        self.server_address= self.config.glove_twitter_27B_25d_path
        self.input_file = self.local_address
        self.output_file = 'glove.twitter.27B.25d.txt.word2vec'
        glove2word2vec(self.input_file, self.output_file)
        self.model=KeyedVectors.load_word2vec_format(self.output_file, binary=False)

    def extend_query(self, query):
        if query==None:
            return None
        result = []
        if type(query)!=list and type(query)==str:
            result.append(query)
        for term in query:
                try:
                    result.extend([term,self.model.most_similar(term)[0][0], self.model.most_similar(term)[1][0]])
                except:
                    result.append(term)
                    continue
        return result
