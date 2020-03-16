import numpy as np
from dataset   import addition
from model     import Seq2Seq
from trainer   import MiniBatchTrainer
from optimizer import Adam
from common    import pickler


def test_addition(model, x_test, t_test):
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
        if i % 250 == 0:
            print('')
            print(f'{question}=')
            print(f'collect: {answer}')
            print(f'answer : {sampled}')
            print(colors['green'] + 'Collect!'     + colors['white'] if sampled == answer \
             else colors['red']   + 'Incollect...' + colors['white'])

    print('--------------------------------------------------')
    print(f'Collect {correct_num} of {len(x_test)} questions.')
    print('--------------------------------------------------')
    print('')
    

wordvec_size = 16
hidden_size  = 128
batch_size   = 128
max_epoch    = 20
max_grad     = 5.0

x, t = addition.load_data('train')
x_test, t_test = addition.load_data('test')
char_to_id, id_to_char = addition.get_vocab()

vocab_size = len(char_to_id)
model = Seq2Seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = MiniBatchTrainer(model, optimizer)

trainer.train(x=x, t=t, epoch=max_epoch, batch_size=batch_size, max_grad=max_grad, \
              when_a_epoch_ended=lambda : test_addition(model, x_test, t_test))

