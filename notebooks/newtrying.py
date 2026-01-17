import os
import torch
import math
import json
import random
import trackio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import datasets
from dataclasses import dataclass
from typing import List, Dict, Any
from torch.utils.data import DataLoader, random_split
from transformers import AutoImageProcessor, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from dinov3_mlp import DinoV3MLP
#split DATASET for training
#before have claude do some optimizations and actual labeling
data_dir="../data"
#subdirectory names become the labels

full_dataset = datasets.ImageFolder(root=data_dir)

# DINOv3 Backbone
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME) #automatically loads form the file path udnerstands inputs and creates concvevrsion puipeline
backbone = AutoModel.from_pretrained(MODEL_NAME)
image_processor_confif = json.loads(image_processor.to_json_string()) #json;.loads turns json into python dict
backbone_config = json.loads(AutoConfig.from_pretrained(MODEL_NAME).to_json_string())
freeze_backbone = True
model = DinoV3MLP(backbone, 3)
# implement focal loss to normalize w class data
