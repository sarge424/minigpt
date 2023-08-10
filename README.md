# minigpt
A character level generative pretrained transformer implemented in python

MiniGPT is a decoder-only model implemented in pytorch.
It was trained on the Tiny Shakespeare dataset (https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and generates the next token based on the previously generated tokens.

To run the pretrained model (10M paramters, 5000 iterations of training), download this repository and run `load.py` to load weights and sample from the model

While the model and dataset used are too small for the model to actually generate sentences that make sense,
It generates sequences that have correct sentence structure as well as Shakespearean English grammar and vocabulary.

A generated sequence (from the sample model included) is shown below:
```
CLARENCE:
The duke hath power to Pomfret, and Warwick,
Will I be reverent to wield my son,
And ten thus my son, and my oath both,
With my lord, farewell! Well, these Plantageneto,
Who bade me of ware deadly wonder divine?

QUEEN ELIZABETH:
Why, she's on the house of Buckolia?

KING RICHARD III:
Bfasting of all death in woe?

QUEEN ELIZABETH:
Reign, now I am not spoke of arch.
But in the bloody arm of heaven,
To frame women live on other shall beguile:
Which not, the steers, dispatch'd, with themselves,
But in the leash of all their fears dreams,
They call them breathed when they do approach:
Are they are but graced their eyes of your country;
Which we can show us as a name,
Dry 'doves yourselves;' and, if you reconciled,
As iclear you high to piery foes, you
We have at all, Edmiose to your friends,
And much cause of your partners
Of your proper tried in summer's hook,
But bear the poor horrow-hear me,
Which doth ave mine honour weeds, drawm in your love?
