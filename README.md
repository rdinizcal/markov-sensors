# Self-Organized Mapping for Environmental Non-determinism Simulation at WEKA

This repository contains the code implementation for the technical titled "Self-Organized Mapping for environmental non-determinism simulation at WEKA".
The goal of the project is to model vital signs from a dataset available online to simulate non-deterministic input for body sensor networks.
 
## Table of Contents

- [Abstract](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running the code](#installation)

 
- results -- contains the documented models, i.e., reprensetation of patients' vital signs using discrete-time markov chains
- code -- contains all the software utilized to derive the model from the inital dataset, i.e., using SOM. 
- report -- contains the technical paper explaninig the method employed and results

## Abstract

Environmental non-determinism demands complex reasoning mechanisms for systems eager to achieve goals on partially-observable and unknown environments. Lately, scientists have been exploring software capable of reorganizing its own internal structure to cope with environmental uncertainties, however it’s not trivial to apply the methods and techniques to developed self-adaptive systems as they may present unpredictable behavior if adaptations are not well validated in design phase. In our study group, we developed a simulation of a Body Sensor Network system with vital signal generation through probabilistic models to simulate environmental non-determinism. In the current study, we apply an one-dimensional self-organized mapping neural network for clustering heart rate data values into ranges that will represent the markov chain states.

## Getting Started

### Prerequisites

The code asks for the following libraries:

 - matplotlib
 - numpy
 - scipy
 - sklearn

### Running the code

```bash
$ cd code/
$ python main.py
```

As output, you should expect a file containing a markov chain represented with a transition matrix in a file named \[vitalsign\]_mc.txt.

## Acknowledgments
Mention and give credit to any individuals, projects, or resources that have contributed to or inspired your work.

## Corresponding Author

 [Ricardo D. Caldas](https://rdinizcal.github.io)
