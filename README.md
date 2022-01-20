# Neural Network Fairness Analysis
Goal: Propagate input distributions though neural networks to prove fairness properties


To run the code, first install requirements:

```
pip3 install -r requirements.txt
```

You also want to install nnenum and put it on your `PYTHONPATH`: `https://github.com/stanleybak/nnenum`


Here's an explantion of the main code files that should run:


*  `python3 fairness_plot.py` - Run reachability analysis to split input into polytopes, and then integrate the probability distributions to get probabilities of outputs (output image: `Male_Seed_0.png`) 
* `python 3 fairness_random_plot.py` - Sample the inputs distribution to get the expected result of exact analysis (output images: `rand_Male_Seed_0_outputs.png` and `rand_Male_Seed_0_inputs.png`)
