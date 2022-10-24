# Text_Summarizer_with_RNN

This is a project for our bigdata ecosystem class. We've developed a Text Summarizer using Recurrent Neural Network and Sequence-to-Sequence model. The data set being used is DeepMind's CNN and Dailymail dataset of news stories.

The dataset stories can be downloaded from [here](https://cs.nyu.edu/~kcho/DMQA/). We used stories dataset from the both.

Our code is inspired from tensorflow's [textsum](https://github.com/tensorflow/models/tree/master/research/textsum).

The processed dataset has been uploaded to dropbox. Data can also be manually processed by running the make_datafiles.py code which first tokenizes the code using stanford nlp PTBtokenizer and then process them into binaries.

The project uses Sequence-to-Sequence attentional model with pointer network. The encoder uses birectional Gated Recurrent Unit(GRU) cells and the decoder uses a unidirectional GRU. We have evaluated our results using ROUGE and have achieved abstractive summaries with ROUGE 1,2 and L scores being 28.88, 10.38, 26.66 respectively.
