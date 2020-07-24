# Algorithms for computing coding and non-coding dinucleotide forces

This repository contains the code to compute dinucleotide forces (for a definition
give a look a [this paper](https://www.pnas.org/content/111/13/5054.short)),
as well as a simple example of its use.

All of the code is written in Python3.

Dependencies required to run the code include `numpy` and `biopython` (for forces in the coding case). 
Moreover an installation of [jupyter](https://jupyter.org) is needed to run the `*.ipynb` notebook.


## Repository structure:
- The `forces_noncoding_multi.py` file containts the script to compute forces
in the non-coding case (without caring about codons); it also contains the
script to get the log-probability of a given sequence according to the 
model.
- The `forces_coding_multi.py` file containts the script to compute forces
in the coding case; it also contains the script to get the log-probability 
of a given sequence according to the model in this case.
- The `forces_example.ipynb` notebook contains a short tutorial of the
usage of the scripts.

