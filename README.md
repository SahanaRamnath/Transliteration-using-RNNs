# Transliteration-using-RNNs
Implementing a Text Transliteration system on the NEWS 2012 English-Hindi Dataset using Tensorflow. Done as an assignment for the course 'Deep Learning : CS7015'

# Dependencies
* Python(used v2.7.12)
* Tensorflow (used v1.10.1)
* numpy (used v1.13.3)
* matplotlib (used v1.5.3)
* pandas (used v0.22.0)

# Dataset
The dataset used is NEWS 2012 (Named Entities Workshop) shared task dataset, containing input words of
different lengths. The training set has 13122 datapoints, validation set has 997 datapoints, test set
(partial) has 400 datapoints and test set (final) has 1000 datapoints.

# Implementation Details
Recurrent neural networks (LSTM) were used for the encoder and the decoder, for the input and output sequences respectively. For encoding the input sequence, we use tensorflow's implementation of bidirectional_dynamic_rnn for the encoder. For predicting the output sequence, we implement a custom decoder with attention mechanism, using basic tensorflow operations rather than tensorflow's seq2seq module. The model is trained end-to-end using a cross entropy loss.
Techniques such as early stopping, dropout, uni/bi-directional encoders and stacked/non-stacked decoders have been experimented with. The observations and conclusions of these experiments, and more specific hyperparameter details+equations can be found in ```report.pdf```.

# Code Organization
* ```train.py``` : Code to train and test the RNN model
* ```train_uni.py``` : Code with unidirectional encoder
* ```create_vocab.py``` : Code to create and save english and hindi vocabulary
* ```plot_loss.py``` : Code to plot loss and accuracy plots
* ```attention_plots.py``` : Code to plot the attention weights on the given test set
* ```report.pdf``` : Detailed report with all experiments, plots and explnations
* ```run.sh``` : Command for running inference with the best hyperparameter configuration

# Attention Plots
The attention weights obtained are shown below for different words in the test set. As can
be seen in the plots below, the implemented attention mechanism works well(almost
perfectly!), even for relatively long sequences. Most of the attention plots have meaningful
character alignments, with only a few characters having the wrong alignments(such as ’NI’ in the
last plot). There no non-contiguous alignments observed. A lot of the characters have perfect one-one or one-many alignments(see ’AU’ in CAULFIELD or ’CO’ in ACORN). Even when the one-many
alignments are not perfect, the higher probability is almost always assigned to the correct english
character(s).
<img src="https://user-images.githubusercontent.com/17588365/66266970-54888300-e849-11e9-9cd2-47e30e52fde7.png" width=600>
<img src="https://user-images.githubusercontent.com/17588365/66267044-2e171780-e84a-11e9-93e9-28dde8b9a5f0.png" width=600>
<img src="https://user-images.githubusercontent.com/17588365/66267060-5ef74c80-e84a-11e9-9394-a4e5bfea014c.png" width=600>
<img src="https://user-images.githubusercontent.com/17588365/66267053-4c7d1300-e84a-11e9-9af4-5eee158ffafa.png" width=600>
<img src="https://user-images.githubusercontent.com/17588365/66267064-72a2b300-e84a-11e9-86a0-5e40fd78d89d.png" width=600>
