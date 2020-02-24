from dataset   import ptb, context
from model     import CBOW
from trainer   import MiniBatchTrainer
from optimizer import Adam
from common    import pickler

window_size = 5
hidden_size = 100
batch_size  = 100
max_epoch   = 10

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)


contexts, target = context.create_contexts_target(corpus, window_size)

model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = MiniBatchTrainer(model, optimizer)

trainer.train(contexts, target, max_epoch, batch_size)

word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pickler.save(params, 'params_cbow')

