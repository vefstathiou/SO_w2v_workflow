import time
from gensim.parsing import *
from gensim.parsing import *
from gensim.models.phrases import *
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

def read_stopwordlist(filename):
    s = open(filename, mode='r', encoding='utf-8-sig').read()
    s.replace('\n','')
    stopwlist = s.split(',')
    return stopwlist


## Memory-friendly iterative versions.
# Iteratively reads sentences from a list of text files.
class Sentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            file_tokens=0
            for line in open(os.path.join(self.dirname, fname)):
                tk=line.split()
                file_tokens=file_tokens+len(tk)
                yield tk
            print('file tokens: ' + str(file_tokens))


# Iteratively cleans out of the given list of sentences, the stopwords given in the stopword list.
class CleanStopwords(object):
    def __init__(self,sentences,stopword_list):
        self.sentenceList = sentences
        self.swords=stopword_list
        self.len_before_total=0
        self.len_after_total=0

    def __iter__(self):
        for s in self.sentenceList:
            len_before=len(s)
            self.len_before_total=self.len_before_total+len_before
            t= [word for word in s if word not in self.swords]
            len_after=len(t)
            self.len_after_total=self.len_after_total+len_after
            yield t
        print('total words before stop word removal: ' + str(self.len_before_total))
        print('total words after stop word removal: '+ str(self.len_after_total))


