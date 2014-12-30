
# In limbo fMRI package 
This Python package is a supplement to the the paper
``An antidote to the imager's fallacy, or how to identify brain areas that are
in limbo''.

It includes the original code, and, importantly, also a collection of
[NiPype](https://github.com/nipy/nipype) interfacs and workflows that should
make it relatively easy to apply the methods of the paper to your own fMRI
dataset.

## Installation
To install this package, simply do

	pip install in_limbo

or for the latest development version:

	pip install git+git://github.com/Gilles86/in_limbo.git#egg=in_limbo

## Simulations
Notebooks redoing the simulations of the paper can be found here
* [Single subject case](http://nbviewer.ipython.org/github/Gilles86/in_limbo/blob/master/notebooks/single_subject.ipynb)
* [Multiple subjects case](http://nbviewer.ipython.org/github/Gilles86/in_limbo/blob/master/notebooks/multiple_subjects.ipynb)


## Examples

* [How to do a level-1 analysis using the sandwich estimator](http://nbviewer.ipython.org/github/Gilles86/in_limbo/blob/master/notebooks/How%20to%20use%20standard%20level%201%20sandwich%20estimator.ipynb)
* [How to do a level-2 analysis including an in limbo-anaysis and using the sandwich estimator](http://nbviewer.ipython.org/github/Gilles86/in_limbo/blob/master/notebooks/How%20to%20use%20standard%20level%202%20sandwich%20estimator.ipynb)
