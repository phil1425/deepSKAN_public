# DeepGTA
Analysis of time resolved spectra using Deep Learning and Global/Target Analysis

## Installation instructions:
- This package is meant to be installed with pip: `pip install .`

## Requirements
The versions specified in `requirements.txt` are known to work, newer versions will probably be fine but have not been tested
- Python 3.6
- keras (using tensorflow backend)
- networkx
- numpy
- scipy
- matplotlib

## Usage
In the `examples` folder, there are three different ways this package can be used:
- Training: Trains a resNet-like CNN on synthetic training data generated on the fly.
- Complete Analysis: Predicts model using neural network, then performs GTA with predicted model.
- Suggestion: gives suggestions about possible pathways. this is implemented as a webinterface.

## Documentation
Basic descriptions of every function are included in their docstring.