# No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems

This code implements the "GEORGE" algorithm from the following paper (to appear in NeurIPS 2020!):

Nimit Sohoni, Jared Dunnmon, Geoffrey Angus, Albert Gu, Christopher Ré

[No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems](https://arxiv.org/abs/2011.12945)

## Abstract

In real-world classification tasks, each class often comprises multiple finer-grained "subclasses."
As the subclass labels are frequently unavailable,
models trained using only the coarser-grained class labels often
exhibit highly variable performance across different subclasses.
This phenomenon, known as _hidden stratification_, has
important consequences for models deployed in safety-critical applications such as medicine.
We propose GEORGE, a method to both measure and mitigate hidden stratification
_even when subclass labels are unknown_.
We first observe that unlabeled subclasses are often separable in the feature space of deep models,
and exploit this fact to estimate subclass labels for the training data via clustering techniques.
We then use these approximate subclass labels as a form of noisy supervision
in a distributionally robust optimization objective.
We theoretically characterize the performance of GEORGE in terms of the worst-case generalization error across any subclass.
We empirically validate GEORGE on a mix of real-world and benchmark image classification datasets, and
show that our approach boosts worst-case subclass accuracy by up to 22 percentage
points compared to standard training techniques, without requiring any information about the subclasses.

## Setup instructions

Prerequisites: Make sure you have Python>=3.6 and PyTorch>=1.5 installed. Then, install dependencies with:
```bash
pip install -r requirements.txt
```

Next, either add the base directory of the repository to your PYTHONPATH, or run:
```bash
pip install -e .
```

## Demo

We provide a simple demo notebook at `tutorials/Basic-Tutorial.ipynb`.
This example can also be run as a script:
```bash
python stratification/demo.py configs/demo_config.json
```

## Configuration options

The first argument to the script should be the path to the configuration file. Default configurations for the GEORGE experiments in the paper are in the `configs/` directory. The configuration can also be modified by directly set config values using the command-line interface. Use `=` to demarcate key-value pairs, and use `.` to access nested dictionaries as specified in the config; for example:
```bash
python stratification/run.py configs/mnist_george_config.json exp_dir=checkpoints/new-experiment classification_config.num_epochs=50
```

The modified config will be saved at `config['exp_dir']`. For the complete configuration definition, see `stratification/utils/schema.py`.
