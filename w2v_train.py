


import time
from gensim.models import Word2Vec
import w2v_utils
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SWORDS_FILEPATH='CoreNLP_swords.txt'
DATA_FILEPATH='data_chunks'

metadata=''
m='starting@: ' + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)
metadata=metadata+m
st_time=time.time()

swordlist=w2v_utils.read_stopwordlist(SWORDS_FILEPATH)

m='starting reading iterator@: ' + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)
metadata=metadata+m

iter_sttime=time.time()
sentences = w2v_utils.Sentences(DATA_FILEPATH)

m='finished reading iterator@: ' + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)
metadata=metadata+m

iter_endtime=time.time()
iter_eltime=iter_endtime-iter_sttime

m='time elapsed for reading iterator: ' + str(iter_eltime) + '\n'
print(m)
metadata=metadata+m

m='starting stop word removal@: ' + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)
metadata=metadata+m

sword_sttime=time.time()
cleanSentences = w2v_utils.CleanStopwords(sentences,swordlist)
sword_endtime=time.time()
sword_eltime=sword_endtime-sword_sttime

m='finished stop word removal@: ' + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)
metadata=metadata+m

m='time elapsed for removing stopwords: ' + str(sword_eltime) + '\n'
print(m)
metadata=metadata+m

m= 'starting word2vec @:'  + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)
metadata=metadata+m

w2v_sttime=time.time()
model = Word2Vec(cleanSentences, size=100, window=5, min_count=5, workers=4,iter=3)
w2v_endtime=time.time()
w2v_eltime=w2v_endtime-w2v_sttime

m='finished w2v@: ' + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)

metadata=metadata+m

m='time elapsed for w2v: ' + str(w2v_eltime) + '\n'
print(m)
metadata=metadata+m

m='started saving model @: ' + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)
metadata=metadata+m

model_sttime=time.time()
model.wv.save_word2vec_format('MDL_toy.txt', binary=False)
model.wv.save_word2vec_format('MDL_toy.bin', binary=True)
model_endtime=time.time()
model_eltime=model_endtime-model_sttime

m='finished saving model@: ' + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)
metadata=metadata+m

m='time elapsed for saving model: ' + str(model_eltime) + '\n'
print(m)
metadata=metadata+m

end_time=time.time()
el_time=end_time-st_time

m='everything finished @: ' + str(time.asctime(time.localtime(time.time()))) + '\n'
print(m)
metadata=metadata+m


m='time elapsed in total: ' + str(el_time) + '\n'
print(m)
metadata=metadata+m

outfile='metadata_toy.txt'
with open (outfile,'w') as fl:
    fl.write(metadata)
