import cv2
import numpy as np
from skimage import feature
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import os
from scipy.io import loadmat
from PIL import Image, ImageFile
import pywt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from dataset import configdataset
from download import download_datasets


def extract_rsift(image):
    """
    Extract Hessian-Affine keypoints and rootSIFT descriptors.
    """
    image = np.array(image.convert('RGB'))
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray_image = cv2.equalizeHist(gray_image)

    # Detect Hessian-Affine keypoints
    sift = cv2.SIFT_create(nfeatures=200)
    # sift = cv2.ORB_create(nfeatures=500)
    # Compute SIFT descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    # print(keypoints)

    # Apply rootSIFT
    if descriptors is not None:
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + 1e-6)
        descriptors = np.sqrt(descriptors)

    return keypoints, descriptors

def compute_smk(descriptors, k_neighbors=5):
    """
    Simulate a simplified Selective Match Kernel (SMK) implementation by aggregating neighbors.
    """
    if descriptors is None:
        return np.zeros(128)  # Placeholder in case of no descriptors

    # Use nearest neighbor search to aggregate descriptors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(descriptors)
    distances, indices = nbrs.kneighbors(descriptors)

    # Aggregate neighboring descriptors (simplified approach)
    aggregated = np.mean([descriptors[idx] for idx in indices], axis=0)
    return aggregated.flatten()

def extract_traditional_features(image, vocabulary):
    _, rsift_descriptors = extract_rsift(image)
    # vlad_features = compute_vlad(rsift_descriptors, vocabulary)
    # return vlad_features
    smk_features = aggregate_descriptors(rsift_descriptors, vocabulary)
    return smk_features

def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def build_vocabulary(all_descriptors, num_clusters=64):
    """
    Create a vocabulary of visual words using KMeans clustering.
    """
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(all_descriptors)
    return kmeans

def compute_vlad(descriptors, vocabulary):
    """
    Compute the VLAD encoding given the descriptors and a visual vocabulary.
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocabulary.n_clusters * descriptors.shape[1])

    # Initialize VLAD vector
    vlad = np.zeros((vocabulary.n_clusters, descriptors.shape[1]))

    # Predict cluster assignments for each descriptor
    predictions = vocabulary.predict(descriptors)

    # Accumulate differences
    for i, cluster in enumerate(predictions):
        vlad[cluster] += (descriptors[i] - vocabulary.cluster_centers_[cluster])

    # Flatten and normalize the VLAD vector
    vlad = vlad.flatten()
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    vlad = vlad / np.linalg.norm(vlad)

    return vlad

def aggregate_descriptors(image_descriptors, vocabulary):
    """
    Aggregates descriptors into a histogram based on the vocabulary.
    """
    # Predict cluster assignments for the descriptors
    predictions = vocabulary.predict(image_descriptors)

    # Initialize histogram
    num_clusters = vocabulary.n_clusters
    histogram = np.zeros(num_clusters)

    # Populate histogram
    for prediction in predictions:
        histogram[prediction] += 1

    # Normalize the histogram
    histogram = histogram / np.linalg.norm(histogram)

    return histogram

def extract_smk_features(image, vocabulary):
    """
    Extracts SMK features for an image given a pre-built vocabulary.
    """
    descriptors = extract_rsift(image)
    if descriptors is not None and len(descriptors) > 0:
        return aggregate_descriptors(descriptors, vocabulary)
    else:
        return np.zeros(vocabulary.n_clusters)

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

    # build vlad vocab
    all_descriptors = []
    for i in np.arange(cfg['nq']):
        qim = pil_loader(cfg['qim_fname'](cfg, i)).crop(cfg['gnd'][i]['bbx'])
        _, rsift_descriptors = extract_rsift(qim)
        print(rsift_descriptors.shape)
        if rsift_descriptors is not None:
            all_descriptors.append(rsift_descriptors)
        print('>> {}: Building vocab {}'.format(test_dataset, i + 1))

    for i in np.arange(cfg['n']):
        im = pil_loader(cfg['im_fname'](cfg, i))
        _, rsift_descriptors = extract_rsift(im)
        if rsift_descriptors is not None:
            all_descriptors.append(rsift_descriptors)
        print('>> {}: Building vocab {}'.format(test_dataset, i + 1))

    all_descriptors = np.vstack(all_descriptors)
    # Build the vocabulary
    num_clusters = 64  # Set the desired number of visual words
    vocabulary = build_vocabulary(all_descriptors, num_clusters)

    # query images
    for i in np.arange(cfg['nq']):
        qim = pil_loader(cfg['qim_fname'](cfg, i)).crop(cfg['gnd'][i]['bbx'])
        ##------------------------------------------------------
        ## Perform image processing here, eg, feature extraction
        ##------------------------------------------------------
        features = extract_traditional_features(qim, vocabulary)
        query_features.append(features)
        print('>> {}: Processing query image {}'.format(test_dataset, i + 1))

    for i in np.arange(cfg['n']):
        im = pil_loader(cfg['im_fname'](cfg, i))
        ##------------------------------------------------------
        ## Perform image processing here, eg, feature extraction
        ##------------------------------------------------------
        features = extract_traditional_features(im, vocabulary)
        data_features.append(features)
        print('>> {}: Processing database image {}'.format(test_dataset, i + 1))

    # query_features = np.nan_to_num(query_features)
    # data_features = np.nan_to_num((data_features))
    features = {
        '__header__': b'Feature data',
        '__version__': '1.0',
        '__globals__': [],
        'Q': np.array(query_features).T,
        'X': np.array(data_features).T
    }

    print(features['Q'].shape)
    print(features['X'].shape)

    sio.savemat('../features/features_with_rsiftsmk.mat', features)

