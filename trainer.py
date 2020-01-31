import numpy as np
from optimizer import SGD
from dataset   import mnist
from model     import TwoLayerNet

max_epoch = 1000
batch_size = 100
hidden_size = 50
learning_rate = 0.1

# load training data.
(x, t), (_x_test, _t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)

model = TwoLayerNet(x.shape[1], hidden_size, t.shape[1])
optimizer = SGD(learning_rate)

data_size = x.shape[0]
max_iters = data_size // batch_size

# make mini-batch.
mini_batch_mask = np.random.choice(x.shape[0], batch_size)
x_batch = x[mini_batch_mask]
t_batch = t[mini_batch_mask]

for epoch in range(max_epoch):
    loss = model.forward(x_batch, t_batch)
    model.backward()
    optimizer.update(model.params, model.grads)

    # print training loss.
    print('| epoch %d | loss %.2f' %(epoch+1, loss))

model.print_loss_state()

