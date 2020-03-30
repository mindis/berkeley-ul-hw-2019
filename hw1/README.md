## UC Berkeley Deep UL Course
## Homework 1

### 1. Warmup

See:

```
warmup.py
```

See figures/1_1 for plots.

The model probabilities align well when plotted against the data.
The sample empirical frequencies are similar in distribution shape but with the 
mean of the peaks slightly misaligned, this offset reduces with more samples.


### 2. Two-Dimensional Data

See:

```
two_dimensional_data.py
mlp_model.py
MADE.py
```

See figures/1_2 for plots.
See logs/1_2 for logs.

Both have empirical distributions that match the training distribution.


### 3. High-Dimensional Data

See:

```
high_dimensional_data.py
pixelCNN.py
```

In the factorised case probabilities are treated independently p(r)p(g)p(b), so mask type A all have centre off 
and B all have it on.

Otherwise (non-factorised) the full joint is as in the paper and uses the full joint p(r,g,b) = p(r)p(g|r)p(b|r,g). It can thus model dependencies between
channels. 

Here A and B masks have centre pixels on/off in different channels to allow this.

In my implementation of pixelCNN-MADE, the NN takes one pixel (all channels) from
pixelCNN as aux input and one pixel of input from data and outputs one pixel
of all channels (with MADE over channels). This way each pixel in pixelCNN can
represent information for all previous pixels. I toyed with a MADE that took all pixels
from input and pixelCNN and masked each output pixel to ensure the conditioning but this didn't
work as well for me. Gradient clipping helped here as did inputting the aux (pixelCNN) inputs to 
each layer not just the input layer.

See figures/1_3 for plots of example pixelCNN masks.
See code for MADE masks that match the paper.
See visualisation of receptive field for pixelCNN in figures/1_3.
See logs/1_3 for logs and samples of pixelCNN and pixelCNNMADE runs.