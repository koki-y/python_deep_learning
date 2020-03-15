import numpy as np
from dataset   import addition
from model     import Seq2Seq
from trainer   import MiniBatchTrainer
from optimizer import Adam
from common    import pickler

wordvec_size = 16
hidden_size  = 128
batch_size   = 128
max_epoch    = 25
max_grad     = 5.0

print('Load addition data.')
x, t = addition.load_data('train')
x_test, t_test = addition.load_data('test')
char_to_id, id_to_char = addition.get_vocab()

print('Create seq2seq model.')
vocab_size = len(char_to_id)
model = Seq2Seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = MiniBatchTrainer(model, optimizer)

print('Start training.')
for epoch in range(max_epoch):
    trainer.train(x, t, 1, batch_size, max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, answer = x_test[[i]], t_test[[i]]
        answer.flatten()
        answer = answer[1:]
        
        sampled = model.generate(question, char_to_id['_'], len(answer))
        if sampled == answer:
            correct_num += 1

    print(f'{correct_num} / {len(x_test)}')
    correct_num = 0


