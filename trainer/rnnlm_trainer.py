from math import log10, ceil
import numpy as np
import matplotlib.pyplot as plt

class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.loss_list = []

    def train(self, xs, ts, epoch=100, batch_size=None, time_size=5, max_grad=None):
        data_size = xs.shape[0]

        max_epoch = epoch
        max_iters = data_size // (batch_size * time_size)

        model, optimizer = self.model, self.optimizer
        self.time_idx = 0

        # use only printing.
        epoch_digit = ceil(log10(max_epoch))
        data_digit  = ceil(log10(data_size))

        for epoch in range(max_epoch):
            total_loss = 0.0
            loss_count = 0
            for i in range(max_iters):
                stop = batch_size * time_size * (i + 1)
                x_batch, t_batch = self.get_batch(xs, ts, batch_size, time_size)

                loss = model.forward(x_batch, t_batch)
                model.backward()
                if max_grad:
                    clip_grads(model.grads, max_grad)
                params, grads = remove_duplicate(model.params, model.grads)
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                if i % 20 == 0:
                    perplexity = np.exp(total_loss / loss_count)
                    self.loss_list.append(perplexity)
                    # print training loss.
                    print(f'| epoch {epoch+1:{epoch_digit}d}' \
                          f'| {stop:{data_digit}d}/{data_size}' \
                          f'| perplexity {perplexity:.2f}')
                    total_loss, loss_count = 0.0, 0

    def get_batch(self, x, t, batch_size, time_size):
        x_batch = np.empty((batch_size, time_size), dtype='i')
        t_batch = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        leap_size = data_size // batch_size
        offsets = [i * leap_size for i in range(batch_size)]

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                x_batch[i, time] = x[(offset + self.time_idx) % data_size]
                t_batch[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1

        return x_batch, t_batch

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.show()

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算

                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                    params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j) 
                if find_flg: break
            if find_flg: break
        if not find_flg: break
    return params, grads
