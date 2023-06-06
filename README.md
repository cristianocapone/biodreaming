# Towards biologically plausible Dreaming and Planning
This repository is the official implementation of *Towards biologically plausible Dreaming and Planning in recurrent spiking networks*.

#### 1. Specification of dependencies

The code is written in `Python 3` and requires the following external dependences to run:

```
matplotlib
numpy
os
tqdm
gym[atari]
```

Each package can be installed through `pip3` (or anaconda) by running:

> pip3 install <name_of_package>

#### 2. Training code

In order to reproduce the data reported in Fig 2C, run:

> python3 PG_pong_dreaming.py -if_dream 0
> python3 PG_pong_dreaming.py -if_dream 1

to accumulate statistics with and without the "dreaming" phase.


#### 3. Visualization code

To plot the results presented in Fig 2C run:

> python3 PG_comparison.py
