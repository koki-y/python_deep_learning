import numpy as np

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


    def train(self, x, t):
        max_epoch = 10000
        batch_size = 100

        model = self.model
        optimizer = self.optimizer

        data_size = x.shape[0]
        max_iters = data_size // batch_size

        for epoch in range(max_epoch):
            # make mini-batch.
            mini_batch_mask = np.random.choice(x.shape[0], batch_size)
            x_batch = x[mini_batch_mask]
            t_batch = t[mini_batch_mask]

            loss = model.forward(x_batch, t_batch)
            model.backward()
            optimizer.update(model.params, model.grads)

            # print training loss.
            print('| epoch %d | loss %.2f' %(epoch+1, loss))

