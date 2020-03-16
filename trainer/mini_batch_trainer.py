from math import log10, ceil
import numpy as np
import matplotlib.pyplot as plt

def do_nothing():
    pass

class MiniBatchTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []

    def train(self, x, t, epoch=100, batch_size=None, max_grad=None, \
              when_a_epoch_ended=do_nothing):
        model, optimizer = self.model, self.optimizer

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
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            total_loss = 0.0
            loss_count = 0
            for i in range(max_iters):
                start, stop = batch_size*i, batch_size*(i+1)
                # make mini-batch.
                mini_batch_mask = range(start, stop)
                x_batch = x[mini_batch_mask]
                t_batch = t[mini_batch_mask]

                loss = model.forward(x_batch, t_batch)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)

                # print training loss.
                if i % 10 == 0:
                    print(f'| epoch {epoch+1:{epoch_digit}d}' \
                          f'| {stop:{data_digit}d}/{data_size}' \
                          f'| loss {loss:.2f}')

                total_loss += loss
                loss_count += 1

            # Store the avarage of the losses
            self.loss_list.append(total_loss / loss_count)
            
            when_a_epoch_ended()

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
    for i, grad in enumerate(grads):
        total = np.sum(grad ** 2)
        total_norm += total
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
