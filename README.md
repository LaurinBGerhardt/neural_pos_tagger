# neural_pos_tagger
A neural POS tagger based on nested LSTMs

First, two LSTMs tage word prefixes and suffixes as their respective inputs, creating a context-aware vector representation for each position. 
The tensors of the prefixes and suffixes are then concatenated and treated as word embeddings. 
Those are then fed to a sentence-level biLSTM. A linear model then takes the biLSTM's output and predicts one of the tags which were present in the training set, or the unknown tag. 

To start training, and to choose the size of the model, do 

```bash
./rnn-train.py trainfile devfile parameterfile --num_epochs=20 --emb_size=100 --char_rnn_size=50 --word_rnn_size=50 --dropout_rate=0.05 --learning_rate=0.1 > basic_info.txt
```

The script will then create and train the model using the test and development sets, and save it in the parameter file. 
Afterwards one can annotate text by doing 

```bash
./rnn_annotate.py parameterfile testfile > annotatedfile
```

All of the train, dev and test files need to look like this:

```
This   DET
is   V
a   DET
sentence   N
.   PUNCT

This   DET
is   V
one   N
as   COMP
well   ADV
!   PUNCT
```
