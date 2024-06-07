So I'm thinking of letting the autoencoder train on the 1.5B tokens and then after maybe 1 epoch we switch to 200M 

Then we got the option to continue train the autoencoder on 1.5B tokens but that would require two sessions running.

* 6 tokens per triangle for safety
* 200M  tokens 50k models 7k labels.
* 1.5 billion tokens  582819 models with 54k labels
* fine-tune it on maybe 10-20k models

Okay, the most reasonable is not to jump to the sky and waste GPU $$, so the 200M seems like a good start.

But I was thinking of testing training the autoencoder on 1.5B and if it takes like 48hrs using 4090, then we might just train on the 200M tokens only.

Okay, the plan now is to train again, but setting "text_condition_cond_drop_prob" to 25%.

This value will mask the text, so instead of "shovel with a wooden handle" it can become "wooden handle" or "shovel" since it will have a 25% change to drop any of the words in the text.
