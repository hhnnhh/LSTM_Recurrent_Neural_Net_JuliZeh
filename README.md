### Ein Flasche Schampus trinken auf die Schlafkeit der Stadt:
# A LSTM Recurrent Neural Net trained on Juli Zeh
A KERAS Long-Short-Term-Memory (LSTM) Recurrent Neural Net (RNN) trained on parts of the book "Spieltrieb" by Juli Zeh

## Source:
Long-Short-Term-Memory RNN trained to Nietzsche data by [KERAS](https://keras.io/examples/lstm_text_generation/)

## Idea: 
Juli Zeh is my favorite German writer and love the idea to be able to generate text that she should have come up with. Therefore I decided built a Juli-Zeh-Text-Robot, based on text she has written. To begin with, I chose one of her earlier books, “Spieltrieb”, published in 2004.

## LSTM RNN
The basic model is a Recurrent Neural Network, a type of networks which can handle sequential data such as text. Long-Short-Term Memory cells are an extension of these networks (Hofreiter ..). They are included to handle a common problem in neural nets, called the *Backpropagation of Error*. For backpropagation, first a training example is propagated through the network, then the difference between the network output and its expected output is calculated. The difference is calculated by the *loss function*, representing its "cost" or error.  The error is now propagated back through the output to the input layer. The weights of the neuron connections are changed depending on their influence on the error. This should result in an approximation to the desired output.
A common LSTM architecture is composed of a cell, which is the memory part of the LSTM unit, and three "regulators", usually called gates, of the flow of information inside the LSTM unit: an input gate, an output gate and a forget gate.
The language model predicts the probability of the next couple of characters in a sentence based on the words and characters the model was trained on. 

## Data: 
According to the Keras developers, the training data set should at least have ~100k characters, but ~1M is better. I chose the book “Spieltrieb” by Juli Zeh and copied a text file of ~180k characters from it, while randomizing chapters. (By randomizing, I hoped to prevent copyright issues when publishing the data, so nobody would be able to steal the book to read it.)

<!---##Getting started: 
First thing was loading the packages, which was awkwardly the biggest problem I encountered when training the NN. After some days of having system shut downs every few minutes I finally decided to deinstall Anaconda and install Miniconda, cleaning the system, reinstalling the IDE (PyCharm) – which was the best decision, because afterwards everything was finally working fine. 
--->
## Data Preprocessing:
1.	First, the text was imported, read and transformed into lowercase, resulting in ~500k characters, including spaces
```python
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
 print('corpus length:', len(text))
#corpus length: 521650
```
2. In the next step, the chars are sorted, listed, counted and enumerated. 

```python
chars = sorted(list(set(text)))
print('total chars:', len(chars))
#total chars: 70
#enumerating the symbols
char_indices = dict((c, i) for i, c in enumerate(chars))
# \n:0, ' ':1, '!':2..'
indices_char = dict((i, c) for i, c in enumerate(chars))
# 0:\n..
```
<!---Chars are: 
['\n', ' ', '!', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '7', '8', '9', ':', ';', '=', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ß', 'à', 'ä', 'é', 'ë', 'ñ', 'ó', 'ö', 'ü', 'ą', 'ć', 'ę', 'ł', 'ń', 'ś', 'š', 'ż', '–', '’', '…', '‹', '›', '\ufeff']
Apparently U+FEFF is a byte order mark, or ‘BOM’, i.e. an encoding specification for UTF formats. It could also be removed, but I decided to ignore it.
--->

3.	In the next step, the text is parsed with a window containing 40 characters, with 3 overlapping characters

```python
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
```

This is what the "sentences" look like now: 
> 'iffe sind sie abgenudelt wie hitparadens', 'e sind sie abgenudelt wie hitparadensong', 'ind sie abgenudelt wie hitparadensongs v', ' sie abgenudelt wie hitparadensongs vom ', 'e abgenudelt wie hitparadensongs vom let', 

In our case, there were 173870 of these "sentences".
The "next_char" probability will be learned and predicted by the model. 
> ['e', 'r', 'h', 'g', 'a', 'a', ' ', ' ', 'd', 'k', ',', 'a', ' ', 'r', 'u', 'r', 'h', 'n', 'i', 'i', 'g', 'c', 'e', 't', 'c', 'n', 'i', 'e', 'e', ' ', 's', 'o', 'a', 't', 'l', 

4. Finally, the data is vectorized into numpy arrays containing booleans, because the model cannot parse text and numpy arrays are faster to process. In a last step, chars are indexed. 

```python
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
```

## the Model
In the original version, the model contains a single layer of LSTM cells with 128 cells and a Dense layer with 'softmax' activation.


```python
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
model.summary()
```

```python
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```
