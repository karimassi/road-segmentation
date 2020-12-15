# Detecting Roads in Aerial Images using Deep Neural Networks

We implement a classifier following a U-Net architecture that outperforms more basic approaches and segments unseen data with high accuracy. Our model was trained using the Pytorch framework on Google's Colab infrastructure. 

## Contents

- The [src](src) folder contains all our models along with helpers for metric and loss functions.
- The [notebooks](notebooks) folder contains interactive `jupyter` notebooks that we used to train our model.
 
## Setup

The following is a list of dependencies used in this study. You can install them using either `pip` or `anaconda`:
- [Pytorch](https://pytorch.org/) 
- [Pytorch learning rate finder](https://github.com/davidtvs/pytorch-lr-finder)
- Matplotlib 
- Numpy 

To run the various notebooks, you will need `jupyter lab` (or `notebook`). Note that we have moved the files dirring development so the imports migth be broken.

## Training instructions

In the `submission` folder, you will find the `run.py` script along with all the necessary helper files it depends on. The script supose that all those files are in the same directory as itself.
A the top of the `run.py` file are defined 3 varriables :
- `load_pretrained_model` which is a boolean that tell the code if we want to load a pretrained model or train a new one from scratch.
- `pretrained_model_path` the path to the archive that contains the pretrained model. If the the previous varriable is set to `False`  then the resulting trained model will be saved at that path.
- `root_data_path` the path to the directory containing all the training and testing data.

When the program is done training or loading the model depending on the value of `load_pretrained_model`, it will creates a directory called `outputs` in which it will writes the ouput of the model of each testing images. Finally then from this directory, it will creates the submission file with the name `submission.csv`.

## Authors

- Karim Assi (karim.assi@epfl.ch)
- Alexandre Pinazza (alexandre.pinazza@epfl.ch)
- Milan Reljin (milan.reljin@epfl.ch)