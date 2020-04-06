import numpy as np
from dataset   import addition
from model     import Seq2Seq
from trainer   import MiniBatchTrainer
from optimizer import Adam
from common    import pickler


def test(model, x_test, t_test, is_x_reversed):
    colors = {'green' : '\033[92m' , 'red' : '\033[91m', 'white' : '\033[0m'}
    correct_num = 0
    for i in range(len(x_test)):
        question, answer = x_test[[i]], t_test[[i]]
        answer = answer.flatten()[1:]
        
        sampled = model.generate(question, char_to_id['_'], len(answer))
        question = ''.join([ id_to_char[index] for index in question.flatten() ])
        sampled  = ''.join([ id_to_char[index] for index in sampled ])
        answer   = ''.join([ id_to_char[index] for index in answer ])
        if sampled == answer:
            correct_num += 1
        if is_x_reversed:
            question = question[::-1]
        if i % 250 == 0:
            if sampled == answer: 
                result = colors['green'] + 'Collect!'     + colors['white']
            else:
                result = colors['red']   + 'Incollect...' + colors['white']
            print('')
            print(f'{question}=')
            print(f'collect: {answer}')
            print(f'answer : {sampled}')
            print(result)

    print('--------------------------------------------------')
    print(f'Collect {correct_num} of {len(x_test)} questions.')
    print('--------------------------------------------------')
    print('')


wordvec_size = 16
hidden_size  = 128
batch_size   = 128
max_epoch    = 25
max_grad     = 5.0

x, t = addition.load_data('train')
x_test, t_test = addition.load_data('test')
# Reverse input data.
x, x_test = x[:, ::-1], x_test[:, ::-1]
is_x_reversed = True

char_to_id, id_to_char = addition.get_vocab()

vocab_size = len(char_to_id)
model = Seq2Seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = MiniBatchTrainer(model, optimizer)

trainer.train(x, t, max_epoch, batch_size, max_grad, \
              when_a_epoch_ended=lambda : test(model, x_test, t_test, is_x_reversed))

pickler.save(model, 'model_seq2seq')
