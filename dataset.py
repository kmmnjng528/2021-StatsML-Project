import os
import csv
import cv2
import dlib
import torch
import random
import skimage
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image, ImageOps
from scipy.spatial import ConvexHull
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def crop_face_image(detector, predictor, image_path):
    # https://stackoverflow.com/questions/46712766/cropping-face-using-dlib-facial-landmarks
    # !wget https://postfiles.pstatic.net/MjAxOTAzMTNfNSAg/MDAxNTUyNDY5OTMyMTEx.PEKYZdmdqJJIJ_QDJ1FElW2ZtcIMplszAUHX0Kw8jBEg.3_dBREe-0RZVBjQpkItuHmMK9yyiO1_AGDriSUpUdiog.JPEG.chandong83/lenna.jpg?type=w773

    # img = dlib.load_rgb_image(image_path)
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)[..., ::-1]

    if not detector(img):
        return img

    rect = detector(img)[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])

    outline = landmarks[[*range(17), *range(26, 16, -1)]]

    Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
    X[X <= 0] = 0
    Y[Y <= 0] = 0
    Y[Y >= 255] = 255
    X[X >= 255] = 255

    cropped_img = np.zeros(img.shape, dtype=np.uint8)
    cropped_img[Y, X] = img[Y, X]

    return cropped_img

class FaceDataset(Dataset):
    def __init__(self, image_label, transforms):
        self.df = image_label
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'

        # image_dir = '.' + self.df.iloc[index, ]['path'][12:]
        image_dir = self.df.iloc[index, ]['path']
        image_id = self.df.iloc[index, ]['fake'].astype(np.int64)

        image = cv2.imread(image_dir, cv2.COLOR_BGR2RGB)
        # image = crop_face_image(detector, predictor, image_dir)
        image = np.array(image)
        # image = np.transpose(image, (1,2,0))
        target = torch.as_tensor(image_id, dtype=torch.long)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        image = image/255.0

        return image, target

class TestDataset(Dataset):
    def __init__(self, image, transforms):
        self.image = image
        self.transforms = transforms

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'

        image_name = self.image[index]
        image_dir = './test/' + image_name
        
        image = cv2.imread(image_dir, cv2.COLOR_BGR2RGB)
        # image = crop_face_image(image)[...,::-1]
        # image = crop_face_image(detector, predictor, image_dir)
        image = np.array(image)
        # image = np.transpose(image, (1,2,0))
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        image = image/255.0
        
        return image_name, image

def get_train_valid_dataloader(options):

    train_df = pd.read_csv(options.data.train)
    train_df['path'] = train_df['path'].map(lambda x : './data' + x[12:])

    train, valid = train_test_split(train_df, test_size=options.data.test_proportions)

    w = options.input_size.height
    h = options.input_size.width

    transforms_train = A.Compose([
        A.Resize(w, h),
        ToTensorV2(),
    ])

    transforms_valid = A.Compose([
        A.Resize(w, h),
        ToTensorV2(),
    ])

    train_dataset = FaceDataset(image_label=train, transforms=transforms_train)
    valid_dataset = FaceDataset(image_label=valid, transforms=transforms_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=options.data.random_split, num_workers=options.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=options.batch_size, shuffle=options.data.random_split, num_workers=options.num_workers)

    return train_dataloader, valid_dataloader, train_dataset, valid_dataset


if __name__ == '__main__':
    from flags import Flags
    options = Flags('configs/VGG.yaml').get()
    train_dataloader, valid_dataloader, train_dataset, valid_dataset = get_train_valid_dataloader(options)
    print(next(iter(train_dataloader)))
    