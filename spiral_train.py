from dataset   import spiral
from model     import TwoLayerNet
from optimizer import SGD
from trainer   import MiniBatchTrainer
from common    import pickler

hidden_size = 10
learning_rate = 1.0

# Load training data.
x, t = spiral.load_data()

# Create mini batch trainer with TwoLayerNet model and SGD optimizer.
model = TwoLayerNet(x.shape[1], hidden_size, t.shape[1])
optimizer = SGD(learning_rate)
trainer = MiniBatchTrainer(model, optimizer)

x_data_size = x.shape[0]
trainer.train(x=x, t=t, epoch=300, batch_size=(x_data_size//10))

# Save trained model and sample data.
pickler.save(model, "model_spiral")
pickler.save(x,     "data_x_spiral")

