from flair.data import Corpus
from flair.datasets import ColumnCorpus

class NERCorpus:
    def __init__(self, corpus_name) -> None:
        self.corpus_name = corpus_name
        
    def create_corpus(self) -> Corpus:


        corpus = ColumnCorpus(data_folder = '{}/'.format(self.corpus_name), 
                            column_format = {0: 'text', 1: 'ner'},
                            train_file = 'train.conll',
                            test_file = 'test.conll',
                            dev_file = 'dev.conll')


        return corpus


