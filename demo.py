'''
USAGE:
python demo.py --config_file configs/ResNet.yaml --image_path samples/fake.jpg 
'''

import os
import random

import cv2
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from psutil import virtual_memory
from flags import Flags
from utils import get_network

def preprocess(image_path):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    image = A.Compose([
        A.Resize(224, 224),
        ToTensorV2(),
    ])(image=image)['image']
    image = image/255.0
    return image


def main(config_file, image_path):

    options = Flags(config_file).get()

    # Set random seed
    random.seed(options.seed)
    np.random.seed(options.seed)
    os.environ["PYTHONHASHSEED"] = str(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed(options.seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    is_cuda = torch.cuda.is_available()
    print("--------------------------------")
    print("Running {} on device {}\nWARNING: THIS IS DEMO MODE!!\n".format(options.network, options.device))

    # Print system environments
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    torch.cuda.empty_cache()
    print(
        "[+] System environments\n",
        "Device: {}\n".format(torch.cuda.get_device_name(current_device)),
        "Random seed : {}\n".format(options.seed),
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    model = get_network(options)
    model.load_state_dict(torch.load(options.checkpoint), strict=False)
    model.to(options.device)
    model.eval()

    print(
        "[+] Network\n",
        "Type: {}\n".format(options.network),
        "Checkpoint: {}\n".format(options.checkpoint),
        "Model parameters: {:,}\n".format(
            sum(p.numel() for p in model.parameters()),
        ),
    )

    summary(model, (3, 224, 224), 32)

    with torch.no_grad():
        classes = ['Real', 'Fake']
        input = preprocess(image_path).to(options.device)
        output = model(input.unsqueeze(0))
        prob = F.softmax(output, dim=1)

        _, preds = output.max(dim=1)
        preds = preds.cpu().item()

        print(
        "[+] Result\n",
        "Image path: {}\n".format(image_path),
        "This image is {}({:.5f})\n".format(classes[preds], prob[0][preds].cpu().item())
        )

if __name__ == '__main__':
    fire.Fire(main)