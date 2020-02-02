from dataset   import mnist
from common    import pickler
from model     import TwoLayerNet
from optimizer import SGD
from trainer   import Trainer

hidden_size = 50
learning_rate = 0.1

# load training data.
(x, t), (_x_test, _t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)
model = TwoLayerNet(x.shape[1], hidden_size, t.shape[1])
optimizer = SGD(learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t)

pickler.save(model)

