from dataset   import mnist
from model     import TwoLayerNet
from optimizer import SGD
from trainer   import Trainer
from common    import pickler

hidden_size = 50
learning_rate = 0.1

# Load training data.
(x, t), (_x_test, _t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)

# Make trainer with TwoLayerNet model and SGD optimizer.
model = TwoLayerNet(x.shape[1], hidden_size, t.shape[1])
optimizer = SGD(learning_rate)
trainer = Trainer(model, optimizer)

trainer.train(x, t)

# Save trained model
pickler.save(model)

