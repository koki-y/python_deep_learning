from dataset   import mnist
from model     import TwoLayerNet
from optimizer import SGD
from trainer   import MiniBatchTrainer
from common    import pickler

hidden_size = 50
learning_rate = 0.1

# Load training data.
(x, t), (_x_test, _t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)

# Create mini batch trainer with TwoLayerNet model and SGD optimizer.
model = TwoLayerNet(x.shape[1], hidden_size, t.shape[1])
optimizer = SGD(learning_rate)
trainer = MiniBatchTrainer(model, optimizer)

x_data_size = x.shape[0]
trainer.train(x=x, t=t, epoch=100, batch_size=(x_data_size//60))

# Save trained model
pickler.save(model)

