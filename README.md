### Ein Flasche Schampus trinken auf die Schlafkeit der Stadt:
# A LSTM Recurrent Neural Net trained on Juli Zeh
A KERAS **Long Short-Term Memory** (LSTM) **Recurrent Neural Net** (RNN) trained on the book "Spieltrieb" by Juli Zeh

Disclaimer: Knowledge of the German language will probably increase the fun. Highly recommended for fans of German dadaism :) 

## Source:
Long Short-Term Memory RNN designed to be trained on Nietzsche, provided by [KERAS](https://keras.io/examples/lstm_text_generation/)

## Idea: 
Juli Zeh is my favorite German writer and love the idea to be able to generate text that she could have come up with. Therefore I decided built a Juli-Zeh-Text-Generator, based on text she has written. To begin with, I chose one of her earlier books, “Spieltrieb”, published in 2004.

## Outline:
1. Data Preprocessing
1. Model Setup
1. Model Training
1. Model Optimization
    1. More Neural Layer
    1. Less Dimensions
    1. More Epochs
        1. integrating *EarlyStopping* Callback function
    1. New activation function (Softgrad replacing Softmax)
    1. Different Optimizers (default = RMSProp)
        1. Adagrad
        1. Adam
    

## LSTM RNN
The basic model is a **Recurrent Neural Network**, a network type which can handle sequential data such as text. Long Short Term Memory (LSTM) cells are an extension of these networks ([Hochreiter & Schmidthuber, 1997](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory/link/5700e75608aea6b7746a0624/download)). They are included to handle a common problem in neural nets, called the *Backpropagation of Error*. 

For backpropagation, first a training example is propagated through the network, then the difference between the network output and its expected output is calculated. The difference is calculated by the **loss function**, representing its *cost* or error.  The error is now propagated back through the output to the input layer. The weights of the neuron connections are changed depending on their influence on the error. This should result in an approximation to the desired output.
A common LSTM architecture is composed of a cell, which is the memory part of the LSTM unit, and three "regulators", usually called gates, of the flow of information inside the LSTM unit: an **input gate**, an **output gate** and a **forget gate**.

The language model predicts the probability of the next couple of characters in a sentence based on the words and characters the model was trained on. 

## Data: 
According to the Keras developers, the training data set should at least have ~100k characters, but ~1M is better. I chose the book “Spieltrieb” by Juli Zeh and copied a text file of 521.548 characters (including spaces) from it, while randomizing chapters. (By randomizing, I hoped to prevent copyright issues when publishing the data.)

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
The **Dense layer** is added to get output in format needed by the user. Here ```Dense(len(chars))``` means 70 different classes of output, i.e. 70 chars, will be generated using *softmax* activation. 
<!---(In case you are using LSTM for time series then you should have Dense(1). So that only one numeric output is given.) --->
Models with LSTM layers automatically run on CPU, if the layers are added directly. 

The **softmax function** (or Maximum Entropy (MaxEnt) Classifier), is an activation function that turns numbers aka logits into probabilities that sum to one. Softmax function outputs a vector that represents the probability distributions of a list of potential outcomes. ([More about softmax.](https://towardsdatascience.com/softmax-function-simplified-714068bf8156))



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

```python
model.fit(x, y,
        batch_size=128,
        epochs=60,
        callbacks=[print_callback])
print(generate_text2(500, 0.2))
```

Final output after two hours of training, with a temperature of 0.2.
With a lower temperature, the probabilities will be preserved - as generated by the model, with higher temperatures, they will be equalized. 

>flasche schampus trinken und sich hoch der stadt schließlich auf dem gewicht seines gesicht verstanden, wie er sich auf dem gesicht zu seiner schlafkeit auf dem schulhofischen stelle, die sich auf dem schlimme der stadt. ada war schon die schlafkeit der frau auf dem stadt und schließlich auf dem gesicht verstanden, die ausgeschauferung der stadt und schließlich auf dem gesicht aus der schlechtliche auf dem gesicht versteckt. das gesicht ergaben sie die stirn starrte, dass sie die schlafkeit auf dem gesicht auf der stelle, sich auf de

With a higher temperature of 1 ```print(generate_text2(500, 1))``` the text starts to sound like being created by German dadaist like Christian Morgenstern (`Der Flügelflagel gaustert/ durchs Wiruwarowolz.') or Hans Sachs. Funny, but nooo.
<!--- train LSTM with dadaist lyrics!--->

>s als geschichte und damit als etwas veranstellt hatte. seinen mit st, die obs rattua, das ewigen musste, so kann sie ruhig, musser, als wärm sie sich am rotgem eskann, und es in die sympfen nichts verbrauch und einmal den erfolgen schweiz. ada konnte sie indesstünden zu erleichtern, alev stießen der mopta im truber dem dumkel tonz und die sein levonten nach ihre gutmokengene sitzen, sich nichts so wurden die terrisierte zukuzeln und letzter war nur müde, während der sokten zu und in der jahr. ein klart. sie brauen prenspringen ins vi


## Optimization

Yes, we could stop now. But there are several options to improve the model: 
<!---
1. adding more LSTM layers      
    - [x] we have one now
1. adding more cells
    - [ ] we have 128, which is quite a lot
1. training with more text 
    - [ ] ~500k characters now
1. using optimizers 
    - [ ] we've been using RMSprop(lr=0.01)
1. training with more epochs 
    - [ ] we have 60
    --->
    
|Optimization options: | in the current version: |
|:--------------------| ---:|
|a) adding more LSTM layers   | 1 layer |
|b) reducing dimensionality of layer |  128 |
|c) run more epochs | 60 |
|d) use the softsign (not softmax) activation function over tanh<sup>1</sup>| softmax|
|more text  | ~500k characters  |
|another optimizer, like RMSProp, AdaGrad or momentum (Nesterovs) | RMSprop(lr=0.01)  |
|evaluate performance at each epoch to know when to stop (early stopping)<sup>1</sup>| --|
|add regularization<sup>2</sup>| --|
|make window larger<sup>2</sup>| 40 char + 3 steps|


<sup>1</sup>More information and optimization methods for LSTM can be found [here](https://pathmind.com/wiki/lstm#long).

<sup>2</sup>Hands-On Machine-Learning with Scikit Learn.. Aurelien Géron, p. 532

### a) adding another LSTM layer
#### "The stutterer" (tro   e       e  dd  ee)
I'll start with adding another LSTM layer. 
It is important to add "return_sequences=TRUE" to the all the LSTM layers ***except*** for the last one. Setting this flag to True lets Keras know that LSTM output should contain all historical generated outputs along with time stamps (3D). Which means, it will contain batch size, time steps, and hidden state. So, next LSTM layer can work further on the data.

```Python
print('Build model...')
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(LSTM(128))
model.add(Dense(len(chars), activation='softmax'))
model.summary()
```
Unfortunately, after training the model for 12.47 hours (!) the results got worse:

diversity: 0.2
>----- Generating with seed: "r direkt über die nackte, inzwischen tro"
r direkt über die nackte, inzwischen tro   e       e  dd  ee         e  ed e d eeee s   e     h e     e  ee    e h      e aeeeed   hd  e    ee d  d  e e e e  ed e eee  e h     de e  eeh  dd ed s  d  ee  er e    ?e eee   d he dee   ee h e      ee h i e e d     e  e     e d     e d ddh   e  ee    e  e     ee  eeeere  h  e ee d  eee e  e  eee   es  d r       dee  d  ee dh d    i    d  e  e  aeie e    e e i   e    e e e e e      hh   d   e 

----- diversity: 1.2
>----- Generating with seed: "r direkt über die nackte, inzwischen tro"
r direkt über die nackte, inzwischen tro vcer lhitlhlnsl rzest hs   sdev tu dree hglieeüisle srihhr tvrie  je uegatehu rtdevhiroiti  bhrhhfenevü ds i e irtlh ehhec giegiuuecddeue ehdrtdniebi i hu aatadhiddeid lre kls heff kggkehgeccwennd eninanhe ueh vhrpiht e hl er s diros dailutss.aehricisgr ere ss tiieeu uieheh ielaocnh te.hriib tnesi.  ädutere lih ik rh urssaetl aiuiswmrliie gaeirdaavdsi re gei krf t h trsevrd.uhaha  oerpepnmveris e

### Two LSTM layers, 32 dimensionality
#### *or* The alliteration machine "schließen stirte schließen schließen schlieben"
The next optimization approach was reducing the dimensionality of the output space to 32. It took only 3.23 hours to train. However, the result is better than the last approach with a dimensionality of 128, but still worse than the first approach with one LSTM layer. 
This model seems to love words with "sch" or "s"? Sounds more like a alliteration machine..
This is what the model looks like:

```python
print('Build model...')
model = Sequential()
model.add(LSTM(32,return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(LSTM(32))
model.add(Dense(len(chars), activation='softmax'))
model.summary()
```

print(generate_text2(500, 0.2))
> nichts ist schöner als überindividuelleit hatte. das schließlichen nicht sich das schließen sich der schlichter der schließen sich die stirte auf der schließen sich das er die schließen sich die schlieben war der schließlichen schließen sich die stierte sich der gescheiden schließen sich der schlichterten der schließen sich die stelle sich die stand und die eine schließer der stiegen und die stelle sich die eine ersten warbelangen und werden sich der schließlichen schließen sich die schließen eine schließen der schließen hatte. die s
>
### only one LSTM layer, trained on 120 epochs
#### The Hans-Arp-would-probably-be-happy Generator

```python
model.fit(x, y,
          batch_size=128,
          epochs=120,
          callbacks=[print_callback])
``` 
After approximately 6 hours and 120 epochs, I do love the lyrics, especially the first sentence ***das luftmus füllt den menschen lungen gegen die stirn.*** Genius! I think, the poet Hans Arp would have loved it. But honestly, the generated text did not get more Juli-Zeh-like after 120 epochs, did it?

----- Generating text after Epoch: 119

```print(generate_text2(500, 0.2))```

>. das luftmus füllt den menschen lungen gegen die stirn. der wasserkommen sich seiner schweigendes nacht eine frau, dass es sich auf den schweigen seiner gesicht zu erreichen. das war er sich auf den tisch einen montern schlag. ada hatte sich auf die schnellen augen und schaute ein bisschen versteckten sich von der schweigende antwort. er war an den ersten atigren spielen der schweigende stelle zu erreichen. das war er die schulzweck auf den konsanten. sie war erfahrt, sich auf die schweigens aus der stirn. wie war es stehen sich auf 

## More epochs, while implementing EarlyStopping

Keras callbacks argument let's us specify a list of objects that Keras will call at the start and end of the training, at the start and end of each epoch (this was already implemented from the beginning, when the lambda-function generated text chunks after each epoch) and even before and after  processing each batch (Géron, 2019, p. 315). 
In the next run, we implement a stopping function ```early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)``` which will interrupt the training, if it does not make any more progress (i.e. when the loss function does not minimize anymore).

Additionally, I implement the TensorBoard, which enables us to visualize the learning curves during training, computation graph, analyze training statistics, view images and much more. It's already integrated into Tensorflow. 

```python
history = model.fit(x, y,
          batch_size=128,
          epochs=100,
          callbacks=[print_callback, tensorboard_cb, early_stopping_cb])
```
So, now let's see, how many epochs will be run if early stopping is integrated? 



### Optimizer: adagrad
### The "the-more-S-the-better" generator 


According to the [Keras](https://keras.io/optimizers/) developers, "Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives, the smaller the learning rate."

It is recommended in the book "Hands-On Machine-Learning" by Aurelién Géron. 

```python
#optimizer used by first keras model:
#optimizer = RMSprop(lr=0.01)
#import keras.optimizers
#from keras.optimizers import adagrad
optimizer = adagrad(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```
With Adagrad, after 2.69 hours, the model seems to infer that "S" is the most important letter in the text.  Is it? No. Actually, it's the fifth frequent character in the text, after the blank space, e, n, i and r. However, this doesn't say anything about the frequencies of the onsets. 

>eine freundin gebraucht hätte. die mädchen er sich war ein der schulen sich der stand sie auf der steiten sich der schleigen sich seine sterten wie ein steiten auf den stiellich der schulen der anderen der für die erteinen auf der steiten sich der steren auf der kannte auf der stand auf der schleisen vor der schleigen besterten geschlenden und der steren sich auf der weile der stand sie sich sie sich sie sich verstien, die schwieden zu schielen, das er sich der schließ und sich sie eine frausen schleigen und der schleisen der schleige
>
### Adam Optimizer
After 3.49 hours

```print(generate_text2(500, 0.2))```

>d vor der sie sich, wie sie jetzt glaubte sie auf den sinn auf der terit, stellte sie auf eine ausiliente sich in der problem sich und schwer zur fenster aus der menschen auf der stelle vor einer schreibung nicht einziger geschlitzten auf der spielerungen straßen, der sich die grund der geschenseite andere nahe hatte sie auf einem kopf auf dem hauben leicht und stempflichen gesehen nicht auf einer sinn auf den steinen zu ungenehnt, und der geschichtsmitzlicht einen ersten fahr. ada den karre zu entragen, sondern das heraus treppen zu 