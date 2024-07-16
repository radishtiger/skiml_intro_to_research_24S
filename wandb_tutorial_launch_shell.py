import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as tr
import torchvision.models as models

from IPython.display import clear_output

from model import MLPNet, ConvNet
from utils import seed_everything, map_dict_to_str, make_loader, run

with open('config.yaml') as f:
    config = yaml.full_load(f)

seed_everything()

my_project_name = "wandb_tutorial_test-launch_shell"
run(config, my_project_name)

