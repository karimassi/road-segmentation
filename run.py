# -*- coding: utf-8 -*-

""" ================ Configuration variables ================ """

# True if you want to load the pretrained model
load_pretrained_model = True
# The path to the pretrained model archive, if previous varriable is False then it will be used to save the final model
pretrained_model_path = "pretrained_model.pkt"
# Path to the data folder (Note the '/' at the end of the string)
root_data_path = "data/"

# General imports
import torch
import os,sys
import numpy as np
import torchvision.io as io 
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import random_split, DataLoader

# Local imports
from src.models.unet import UNet
from src.image_mask_dataset import ImageMaskDataset, FullSubmissionImageDataset
from src.training import train
from src.scripts.mask_to_submission import masks_to_submission


""" ================ Environment set-up ================ """

torch.manual_seed(202042)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of training epochs
total_iterations = 90
learning_rate = 1e-4 

NUM_CHANNELS = 3
NUM_FILTERS = 64

model = UNet(NUM_CHANNELS, NUM_FILTERS).to(device)


""" ================ Model training ================ """

if load_pretrained_model:
    # Load the pretrained model
    if os.path.exists(pretrained_model_path):
        print("Loading model from " + pretrained_model_path)
        state_dicts = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state_dicts['model_state_dict'])
    else :
        print("Unable to load model from " + pretrained_model_path)
        sys.exit(1)
else:
    # Load data
    image_dir = root_data_path + "training/images/"
    gt_dir = root_data_path + "training/groundtruth/"

    dataset = ImageMaskDataset(image_dir, gt_dir)

    # Perform data augmentation: rotation and shearing
    angles = [15, -10, 45, -60, 78]
    for angle in angles:
        rotation = lambda img: TF.rotate(img, angle)
        dataset += ImageMaskDataset(image_dir, gt_dir, rotation)

    shears = [[15, 20], [10, 30], [30, -17], [-3, 20], [-5, -10]]
    for shear in shears:
        transformation = lambda img: TF.affine(img, angle=0, scale=1.0, translate=[0, 0], shear=shear)
        dataset += ImageMaskDataset(image_dir, gt_dir, transformation)

    batch_size = 5

    # Split the data into training and validation
    data_len = len(dataset)
    train_len = int(data_len * 0.8)
    test_len = int(data_len * 0.2)

    dataset_train, dataset_test = random_split(dataset, [train_len, test_len])

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize optimizer and criterion
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model for the total number of epochs
    train(model, criterion, dataloader_train, dataloader_test, optimizer, num_epochs = total_iterations)

    # Save the trained model 
    torch.save(
        { 'model_state_dict': model.state_dict() },
        pretrained_model_path
    )

""" ================ Prediction on testing images ================ """

test_dir = root_data_path + "test_set_images/"

submission_dataloader = DataLoader(
    FullSubmissionImageDataset(test_dir),
    batch_size=1
)

model.eval()
toPIL = transforms.ToPILImage()

output_dir = "outputs"

if output_dir not in os.listdir():
    os.makedirs(output_dir)

for indexes, images in submission_dataloader:
    out = model(images.to(device)).view(2, 608, 608).cpu()
    toPIL(out[0]).save(output_dir + "/file_{:03d}.png".format(indexes.view(-1).item()))


masks_to_submission("submission.csv", *[output_dir + "/" + f for f in sorted(os.listdir(output_dir))])
