## UC Berkeley Deep UL Course
## Homework 1

### 1. Warmup

See `warmup.py`

Results:

         0: Train:      6.644, Val:      6.524
       100: Train:      5.823, Val:      5.854
       200: Train:      5.823, Val:      5.856
       Test set logprob: 5.832
       
See figures/ for plots.

The model probabilities align well when plotted against the data.
The sample empirical frequencies are similar in distribution shape but with the 
mean of the peaks slightly misaligned, this offset reduces with more samples.


### 2. Two-Dimensional Data

See

```
two_dimensional_data.py
mlp_model.py
MADE.py
```