import torch
import torch.nn as nn
import os 
from os.path import join as jn 
from PIL import Image 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms 
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='UNet implementation for green fluorecence segmentation')
# parser.add_argument('--path_images', type=str,default="",help='Path to the images used to train the model')
parser.add_argument('--path_dataset', type=str,default="", help='Path to the dataset used to train the model')
parser.add_argument("--batch_size",type=int,default=32, help="dimensions in which the dataset will be subdivided")
parser.add_argument("--epochs",type=int,default=10, help="number of iterations to train the model for")

# parser.add_argument("--subsampleNo", type=int,help='')
# parser.add_argument('--lr', default=0.1, help='')
# parser.add_argument('--batch_size', type=int, default=512, help='')
# parser.add_argument('--num_workers', type=int, default=0, help='')
try:
    args = parser.parse_args()
except:
    args = None

PATH = args.path_dataset if args is not None else r"C:\Users\Uriel\Desktop\fluorecence_Aug19_dataset\flourecence_20x\dataset"
PATH_IMAGES_TRAIN = jn(PATH,"train","origs")
PATH_MASKS_TRAIN = jn(PATH,"train","masks")

PATH_IMAGES_TEST = jn(PATH,"test","origs")
PATH_MASKS_TEST = jn(PATH,"test","masks")

LEARNING_RATE = 3e-4
BATCH_SIZE = args.batch_size if args is not None else 2
EPOCHS = args.epochs if args is not None else 50
device = "cuda" if torch.cuda.is_available() else "cpu"

