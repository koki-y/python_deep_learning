from math import log10, ceil
import numpy as np

class MiniBatchTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


    def train(self, x, t, epoch=100, batch_size=None):
        model = self.model
        optimizer = self.optimizer

        data_size = x.shape[0]
        max_iters = data_size // batch_size

        max_epoch = epoch
        if not batch_size or batch_size > data_size:
            batch_size = data_size

        # use only printing.
        epoch_digit = ceil(log10(max_epoch))
        data_digit  = ceil(log10(data_size))

        for epoch in range(max_epoch):
            # Shuffle training data.
            idx = np.random.permutation(data_size)
            x = x[idx]
            t = t[idx]

            for i in range(max_iters):
                start, stop = batch_size*i, batch_size*(i+1)
                # make mini-batch.
                mini_batch_mask = range(start, stop)
                x_batch = x[mini_batch_mask]
                t_batch = t[mini_batch_mask]

                loss = model.forward(x_batch, t_batch)
                model.backward()
                optimizer.update(model.params, model.grads)

                # print training loss.
                print(f'| epoch {epoch+1:{epoch_digit}d}' \
                      f'| {stop:{data_digit}d}/{data_size}' \
                      f'| loss {loss:.2f}')

