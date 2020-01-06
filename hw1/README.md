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

Both seem to overfit. 
But both have empirical distributions that match the training distribution.


### 3. High-Dimensional Data

See:

```
high_dimensional_data.py
pixelCNN.py
```

We have two types of mask A and B and also two ways to mask factorised or full.

In the factorised case probabilities are treated independently p(r)p(g)p(b), so mask type A all have centre off 
and B all have it on.

Otherwise, the full joint is as in the paper and uses the full joint p(r,g,b) = p(r)p(g|r)p(b|r,g). It can thus model dependencies between
channels. Here A and B masks have centre pixels on/off in different channels to allow this.

I found the both to work on my debugging cases of a checkerboard or single digit image. But on the full 
dataset, it did not perform as well as the factorised case. I think perhaps this is because it is harder to 
learn the full joint than additionally assuming factorised (independent) channels. The homework says to use 
the factorised method so it is a safe assumption, but I am interested to learn more as to why this is the case.