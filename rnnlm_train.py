import numpy as np
from dataset   import ptb
from model     import SimpleLstm
from trainer   import RnnlmTrainer
from optimizer import SGD
from common    import pickler

def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('evaluating perplexity ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]
        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss
    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl

batch_size   = 20
wordvec_size = 100
hidden_size  = 100
time_size    = 35
lr           = 20.0
max_epoch    = 4
max_grad     = 0.25

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')

vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]
ts = corpus[1:]

model = SimpleLstm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.train(xs, ts, max_epoch, batch_size, time_size, max_grad)
# trainer.plot()

model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print(f'test perplexity:{ppl_test}')

model.reset_state()
pickler.save(model, 'model_lstm')

