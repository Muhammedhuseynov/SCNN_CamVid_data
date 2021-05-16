import os

from transes import get_validation_augmentation, get_preprocessing, get_training_augmentation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import segmentation_models_pytorch as smp

ENCODER = 'vgg16'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class Dataset(BaseDataset):
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    # if dont remove "back", "your_class" replace with your class names
    # CLASSES = ["back","your_class","your_class" ..... ]
    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            data_channel=3,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.data_chan = data_channel
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # print(self.images_fps[i])
        # read data
        if self.data_chan == 3:
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        elif self.data_chan == 1:
            image = cv2.imread(self.images_fps[i], 0)
            image = np.expand_dims(image, axis=2)
        else:
            print("Invalid data channel! Try Again!")
            exit()
        mask = cv2.imread(self.masks_fps[i], 0)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.data_chan == 1:
            image = image.transpose(2, 0, 1).astype('float32')
            mask = mask.transpose(2, 0, 1).astype('float32')

        return image, mask

    def __len__(self):
        return len(self.ids)



def getLoaders(DATA_DIR,CLASSES,data_channel,preproc=None):


    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        data_channel=data_channel,
        preprocessing=get_preprocessing(preprocessing_fn) if preproc is not False else None,
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        data_channel=data_channel,
        preprocessing=get_preprocessing(preprocessing_fn) if preproc is not False else None,
        classes=CLASSES,
    )

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        data_channel=data_channel,
        preprocessing=get_preprocessing(preprocessing_fn) if preproc is not False else None,
        classes=CLASSES,
    )
    test_dataloader = DataLoader(test_dataset)
    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir,data_channel=data_channel,
        classes=CLASSES,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=0)
    # ,test_dataset,test_dataset_vis
    return train_loader,valid_loader,test_dataset,test_dataset_vis









