# Repo for project "Deep Learning for the Study of Adaptive Introgression between the Denisovan Population and the Ancestors of Modern Tibetans"

User guide:

In simulations folder you can find 5 files, for current version run 2 of them in the following order:

1. msprime_burnin.py -- ancestry
2. slim model -- introgression & selection simulation

For the dataset run these files:

1. data_generation.py -- main generator of simulations
2. preprocessing.py -- transfer generated data to csv file
3. processing.py -- creating ml-ready dataset with train, validation and test

The full model is in model folder.
