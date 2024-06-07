So I'm thinking of letting the autoencoder train on the 1.5B tokens and then after maybe 1 epoch we switch to 200M 

Then we got the option to continue train the autoencoder on 1.5B tokens but that would require two sessions running.

* 6 tokens per triangle for safety
* 200M  tokens 50k models 7k labels.
* 1.5 billion tokens  582819 models with 54k labels
* fine-tune it on maybe 10-20k models
