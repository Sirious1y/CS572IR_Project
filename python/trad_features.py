import cv2
import numpy as np
from skimage import feature
import scipy.io as sio
from os import listdir
from os.path import isfile, join

import os
from scipy.io import loadmat

from PIL import Image, ImageFile

from dataset import configdataset
from download import download_datasets



# data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'CS572_IR_Project')
# test_dataset = 'roxford5k'
# print(data_root)
#
# features = loadmat(os.path.join(data_root, '{}_resnet_rsfm120k_gem.mat'.format(test_dataset)))
#
# print(len(features))
# print(features['Q'].shape)
# print(features['X'].shape)

def extract_color_histogram(image):
    image = np.array(image)  # Convert PIL image to a NumPy array
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_hu_moments(image):
    image = np.array(image.convert('L'))  # Convert to grayscale NumPy array
    moments = cv2.moments(image)
    huMoments = cv2.HuMoments(moments)
    return -np.sign(huMoments) * np.log10(np.abs(huMoments))

def extract_lbp_features(image):
    image = np.array(image.convert('L'))  # Convert to grayscale NumPy array
    lbp = feature.local_binary_pattern(image, P=24, R=3, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

if __name__ == '__main__':
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    # Check, and, if necessary, download test data (Oxford and Pairs) and revisited annotation
    # download_datasets(data_root)

    # Set test dataset: roxford5k | rparis6k
    test_dataset = 'roxford5k'

    # ---------------------------------------------------------------------
    # Read images
    # ---------------------------------------------------------------------

    print('>> {}: Processing test dataset...'.format(test_dataset))
    # config file for the dataset
    # separates query image list from database image list, if revisited protocol used
    cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))

    query_features = []
    data_features = []

    # query images
    for i in np.arange(cfg['nq']):
        qim = pil_loader(cfg['qim_fname'](cfg, i)).crop(cfg['gnd'][i]['bbx'])
        ##------------------------------------------------------
        ## Perform image processing here, eg, feature extraction
        ##------------------------------------------------------
        color_hist = extract_color_histogram(qim)
        hu_moments = extract_hu_moments(qim)
        lbp_hist = extract_lbp_features(qim)
        features = np.concatenate((color_hist, hu_moments.flatten(), lbp_hist))
        query_features.append(features)
        print('>> {}: Processing query image {}'.format(test_dataset, i + 1))

    for i in np.arange(cfg['n']):
        im = pil_loader(cfg['im_fname'](cfg, i))
        ##------------------------------------------------------
        ## Perform image processing here, eg, feature extraction
        ##------------------------------------------------------
        color_hist = extract_color_histogram(im)
        hu_moments = extract_hu_moments(im)
        lbp_hist = extract_lbp_features(im)
        features = np.concatenate((color_hist, hu_moments.flatten(), lbp_hist))
        data_features.append(features)
        print('>> {}: Processing database image {}'.format(test_dataset, i + 1))

    features = {
        '__header__': b'Feature data',
        '__version__': '1.0',
        '__globals__': [],
        'Q': np.array(query_features).T,
        'X': np.array(data_features).T
    }

    print(features['Q'].shape)
    print(features['X'].shape)

    sio.savemat('../features/features.mat', features)

