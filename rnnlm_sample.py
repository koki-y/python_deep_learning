import numpy as np
from dataset   import ptb
from model     import SimpleLstm
from trainer   import RnnlmTrainer
from optimizer import SGD
from common    import pickler

batch_size   = 20
wordvec_size = 100
hidden_size  = 100
time_size    = 35
lr           = 20.0
max_epoch    = 4
max_grad     = 0.25

corpus, word_to_id, id_to_word = ptb.load_data('train')

vocab_size = int(max(corpus) + 1)
xs = corpus
ts = corpus

model = SimpleLstm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.train(xs, ts, max_epoch, batch_size, time_size, max_grad)
trainer.plot()

pickler.save(model, 'model_lstm')

