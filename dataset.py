import os
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import albumentations as albu
import segmentation_models_pytorch as smp

def get_loader(ENCODER, ENCODER_WEIGHTS, batch_size, train=False, fold=0):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    image_dataset = Dataset(
        train=train,
        fold=fold,
        augmentation=get_training_augmentation() if train else get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=train, num_workers=16, drop_last=train)
    return loader

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=256, width=256, always_apply=True),
        # albu.RandomCrop(height=320, width=320, always_apply=True),
        # albu.RandomResizedCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            train=True,
            fold=0,
            augmentation=None,
            preprocessing=None,
    ):
        data_dir = osp.join(osp.dirname(__file__), 'dataset')
        file_list = [np.loadtxt(osp.join(data_dir, '{}.txt'.format(i)), dtype=str) for i in range(5)]
        if train:
            file_list = np.concatenate([file_list[i] for i in range(5) if i != fold])
        else:
            file_list = np.array(file_list[fold])
        print('loading total {} images from {} dataset...'.format(len(file_list), 'train' if train else 'val'))
        self.masks_fps = [os.path.join(data_dir, 'gt', file_name) for file_name in file_list]
        self.images_fps = [os.path.join(data_dir, 'data', file_name) for file_name in file_list]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)[:, :, np.newaxis] / 255

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.masks_fps)