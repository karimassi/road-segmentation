# Detecting Roads in Aerial Images using Deep Neural Networks

We implement a classifier following a U-Net architecture that outperforms more basic approaches and segments unseen data with high accuracy. Our model was trained using the Pytorch framework on Google's Colab infrastructure. 

## Contents

- The [src](src) folder contains all our models along with helpers for metric and loss functions.
- The notebooks used to train [CNN](CNN.ipynb) and [UNet](Unet.ipynb). 
- A [run](run.py) script to reproduce the results of our best AICrowd submission (more on that below). 
- A pre-trained model `pretrained_model.pkt`.
 
## Setup

The following is a list of dependencies used in this study. You can install them using either `pip` or `anaconda`:
- [Pytorch](https://pytorch.org/), version 1.7.0
- [Pytorch learning rate finder](https://github.com/davidtvs/pytorch-lr-finder), version 0.2.1
- [Matplotlib](https://matplotlib.org/), version 3.2.2
- [Numpy](https://numpy.org/), version 1.18.5
- [Scikit-image](https://scikit-image.org/), version 0.17.2
To run the various notebooks, you will need `jupyter lab` (or `notebook`). Note that we have moved the files dirring development so the imports migth be broken.

## Training instructions

In the root folder, you will find the `run.py` script. A the top of the file, there are 3 varriables :
- `load_pretrained_model`, a boolean that specifies whether to load a pretrained model or train a new one from scratch (default is True).
- `pretrained_model_path`, the path to the pretrained model. If the the previous varriable is set to `False`  then the resulting trained model will be saved at that path.
- `root_data_path`, the path to the directory containing all the training and testing data.

When the program is done training (or loading the model), it writes, in a `outputs` directory, the prediction masks of our model on the testing images. Using this directory, it creates the submission file `submission.csv`.

## Authors

- Karim Assi (karim.assi@epfl.ch)
- Alexandre Pinazza (alexandre.pinazza@epfl.ch)
- Milan Reljin (milan.reljin@epfl.ch)
