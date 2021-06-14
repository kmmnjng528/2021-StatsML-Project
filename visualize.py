#!/usr/bin/env python
# coding: utf-8

'''
Author:   Kazuto Nakashima
URL:      https://github.com/kazuto1011/grad-cam-pytorch
USAGE:    python visualize.py --config_file configs/VGG.yaml --target_layer=features.28
USAGE:    python visualize.py --config_file configs/ResNet.yaml --target_layer=layer4
USAGE:    python visualize.py --config_file configs/DenseNet.yaml --target_layer=features
USAGE:    python visualize.py --config_file configs/EfficientNet.yaml --target_layer=_conv_head
'''

from __future__ import print_function

import os
import copy
import random
import warnings
import fire
import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from checkpoint import load_checkpoint
from flags import Flags

from utils import (
    get_network,
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def preprocess(image_path):
    raw_image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = A.Compose([
        A.Resize(224, 224),
        ToTensorV2(),
    ])(image=raw_image)['image']
    image = image/255.0
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

def main(target_layer, config_file, topk=1):
    """
    Visualize model responses given multiple images
    """

    options = Flags(config_file).get()

    # Synset words
    # classes = get_classtable()
    classes = ['Fake', 'Real']

    # Model from torchvision
    
    model = get_network(options)
    arch=options.network
    image_paths=options.experiment.vis_input
    output_dir=options.experiment.vis_output

    checkpoint = load_checkpoint(options.test_checkpoint, cuda=True)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint["epoch"]
    train_accuracy = checkpoint["train_accuracy"]
    train_recall = checkpoint["train_recall"]
    train_precision = checkpoint["train_precision"]
    train_losses = checkpoint["train_losses"]
    valid_accuracy = checkpoint["valid_accuracy"]
    valid_recall = checkpoint["valid_recall"]
    valid_precision = checkpoint["valid_precision"]
    valid_losses = checkpoint["valid_losses"]
    learning_rates = checkpoint["lr"]
    model.to(options.device)
    model.eval()

    # summary(model, (3, 224, 224), 32)
    print(model)

    # Images
    images, raw_images = load_images([image_paths])
    images = torch.stack(images).to(options.device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted

    for i in range(topk):
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-vanilla-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    # Remove all the hook function in the "model"
    bp.remove_hook()

    # =========================================================================
    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-deconvnet-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            save_gradient(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

            # Grad-CAM
            save_gradcam(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            save_gradient(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-guided_gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )


if __name__ == "__main__":
    fire.Fire(main)