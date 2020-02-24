from dataset   import context
from model     import SimpleCBOW
from trainer   import MiniBatchTrainer
from optimizer import Adam
from common    import pickler

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

contexts, target = context.load_data(window_size)
vocab_size = contexts.shape[2]

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = MiniBatchTrainer(model, optimizer)

trainer.train(contexts, target, max_epoch, batch_size)
trainer.plot()

pickler.save(model, 'model_cbow')

