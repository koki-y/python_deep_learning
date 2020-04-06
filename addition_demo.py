import numpy as np
import time
from dataset import addition
from common  import pickler
colors = {'green' : '\033[92m' , 'red' : '\033[91m', 'white' : '\033[0m'}

trained_model = pickler.load('model_seq2seq')
char_to_id, id_to_char = addition.get_vocab()

while True:
    print('$', end="")
    s = input()

    if s == 'exit':
        exit()

    if not '+' in s:
        print('must include "+".')
        continue

    num1, num2 = s.split('+')
    if len(num1) > 3 or len(num2) > 3:
        print('the number must have 3 or less digit.')
        continue

    collect = str(int(num1) + int(num2))

    s_adjusted = reversed(s.ljust(7, ' '))
    millisec = int(round(time.time()*1000))
    question = np.array([ char_to_id[c] for c in s_adjusted ], np.int32).reshape(1, 7)
    sampled  = trained_model.generate(question, char_to_id['_'], len(collect))
    answer   = ''.join([ id_to_char[i] for i in sampled ])

    if answer == collect:
        result = colors['green'] + 'Collect!'     + colors['white']
    else:
        result = colors['red']   + 'Incollect...' + colors['white']     
        
    print(f"collect: {collect}")
    print(f"answer : {answer}")
    print(f"{result}")

