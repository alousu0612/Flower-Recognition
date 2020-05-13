import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from random import shuffle
from tqdm import tqdm
import time
import zipfile
from skimage.transform import resize
from sklearn.model_selection import train_test_split


train_dir = './data/train'
test_dir = './data/test'

img_size = 256


def label_folder(folders):
    if folders == 'daisy':
        return 0
    elif folders == 'dandelion':
        return 1
    elif folders == 'rose':
        return 2
    elif folders == 'sunflower':
        return 3
    elif folders == 'tulip':
        return 4


def create_train_data():
    training_data = []
    dirs = os.listdir(train_dir)
    for folders in dirs:
        label = label_folder(folders)
        req_train_dir = os.path.join(train_dir, folders)
        for img in tqdm(os.listdir(req_train_dir)):
            path = os.path.join(req_train_dir, img)
            if (cv2.imread(path)) is not None:
                img = cv2.imread(path)
            img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_CUBIC)
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    train_images, val_images = train_test_split(training_data, )
    np.save('./data/flower_train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]  # image id
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('./data/flower_test_data.npy', testing_data)
    return testing_data


train_data = create_train_data()
test_data = process_test_data()
