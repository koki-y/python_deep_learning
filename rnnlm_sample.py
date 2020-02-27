import numpy as np
from dataset   import ptb
from model     import SimpleRnnlm
from trainer   import RnnlmTrainer
from optimizer import SGD
from common    import pickler

batch_size   = 10
wordvec_size = 100
hidden_size  = 100
time_size    = 5
lr           = 0.1
max_epoch    = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
# Minimize the train data for testing.
corpus_size = 1000
corpus = corpus[:corpus_size]

vocab_size = int(max(corpus) + 1)
xs = corpus
ts = corpus

model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.train(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()

pickler.save(model, 'model_rnnlm')

