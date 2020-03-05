from __future__ import print_function
from keras.callbacks import LambdaCallback
import keras.callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import pandas as pd
import os
import time
# loading the data
os. getcwd()

path = './data/zeh.csv'
# reading the text as utf-8 and removing all capitals
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))
# resulting in corpus length: 521650

chars = sorted(list(set(text)))
print('total chars:', len(chars))
#total chars: 70
#enumerating the symbols
char_indices = dict((c, i) for i, c in enumerate(chars))
# \n:0, ' ':1, '!':2..'
indices_char = dict((i, c) for i, c in enumerate(chars))
# 0:\n..

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
#nb sequences are 173870

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)


for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
#second layer added to improve outcome of first version
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
model.summary()

import keras.optimizers
#from keras.optimizers import adagrad
optimizer = RMSprop(lr=0.01)
#from keras import optimizers
#optimizer = adagrad(lr=0.01)
#from keras.optimizers import adam
#optimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
#import time --> how long does the model run?
t = time.time()
# print_callback --> prints text output at the end of each epoch
# new: tensorboard output --> logging results, creating event files, writing summaries of event files, accessible with GUI
#import os
os. getcwd()
root_logdir = os.path.join(os.curdir, "LSTM_logs")

# import time
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d_-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
# new: stop function, stops when loss does not make any progress
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(x, y,
          batch_size=128,
          epochs=100,
          callbacks=[print_callback, tensorboard_cb, early_stopping_cb])

# how long did it take?
elapsed = time.time() - t
hours=(elapsed/60)/60
print(hours)

## saving 'optimized' models for later
#model.save('julizeh_model.h5')
#model.save('julizeh_model_extraLSTM.h5')
#model.save('juli_zeh_model_extraLSTM_32hiddenlayer.h5')
#model.save('juli_zeh_model_120epochs.h5')
#model.save('juli_zeh_model_adagrad.h5')


# This part can generate text without a seed, which, in my opinion, comes out much nicer.

def generate_text2(length, diversity):
    # Get random starting text
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    for i in range(length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
    return generated

print(generate_text2(500, 0.2))

