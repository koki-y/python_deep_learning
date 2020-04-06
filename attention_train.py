import numpy as np
from dataset   import date
from model     import AttentionSeq2Seq
from trainer   import MiniBatchTrainer
from optimizer import Adam
from common    import pickler
import demo

def test(model, x_test, t_test, is_x_reversed=False):
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
            if is_x_reversed:
                question = question[::-1]
            print('')
            print(f'{question}=')
            print(f'correct: {answer}')
            print(f'answer : {sampled}')
            print(colors['green'] + 'Correct!'     + colors['white'] if sampled == answer \
             else colors['red']   + 'Incorrect...' + colors['white'])

    print('--------------------------------------------------')
    print(f'Correct {correct_num} of {len(x_test)} questions.')
    print('--------------------------------------------------')
    print('')
    


# Load training data.
x, t = date.load_data('train')
x_test, t_test = date.load_data('test')
## Reverse input data.
x, x_test = x[:, ::-1], x_test[:, ::-1]
is_x_reversed = True

char_to_id, id_to_char = date.get_vocab()

# Hyper parameters
wordvec_size = 16
hidden_size  = 256
batch_size   = 128
max_epoch    = 10
max_grad     = 5.0
vocab_size = len(char_to_id)

# Load trained model.
model = pickler.load('model_attention')

if not model:
    model = AttentionSeq2Seq(vocab_size, wordvec_size, hidden_size)
    optimizer = Adam()
    trainer = MiniBatchTrainer(model, optimizer)

    trainer.train(x, t, max_epoch, batch_size, max_grad, \
                  when_a_epoch_ended = lambda : test(model, x_test, t_test, is_x_reversed))

    pickler.save(model, 'model_attention')
else:
    test(model, x_test, t_test, is_x_reversed)

for _ in range(5):
    idx = [ np.random.randint(0, len(x_test)) ]
    x = x_test[idx]
    t = t_test[idx]

    model.forward(x, t)
    d = model.decoder.attention.weights
    d = np.array(d)
    attention_map = d.reshape(d.shape[0], d.shape[2])

    #Reverse for pring
    if is_x_reversed:
        attention_map = attention_map[:, ::-1]
        x = x[:, ::-1]

    row_labels    = [ id_to_char[i] for i in x[0] ]
    column_labels = [ id_to_char[i] for i in t[0] ]
    column_labels = column_labels[1:]

    demo.visualize(attention_map, row_labels, column_labels)
