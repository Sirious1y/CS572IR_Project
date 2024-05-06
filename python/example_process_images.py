# EXAMPLE_PROCESS_IMAGES  Code to read and process images for ROxford and RParis datasets.
# Revisited protocol requires query images to be removed from the database, and cropped prior to any processing.
# This code makes sure the protocol is strictly followed.
#
# More details about the revisited annotation and evaluation can be found in:
# Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking, CVPR 2018
#
# Authors: Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., 2018
import urllib.request
import os
import numpy as np

from PIL import Image, ImageFile

from dataset import configdataset
from download import download_datasets
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from scipy.io import savemat
import numpy as np
import os
from torchvision.models import resnet50, ResNet50_Weights

#---------------------------------------------------------------------
# Set data folder and testing parameters
#---------------------------------------------------------------------
# Set data folder, change if you have downloaded the data somewhere else
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
# Check, and, if necessary, download test data (Oxford and Pairs) and revisited annotation
download_datasets(data_root)

# Set test dataset: roxford5k | rparis6k
test_dataset = 'roxford5k'

#---------------------------------------------------------------------
# Read images
#---------------------------------------------------------------------

def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning 
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

print('>> {}: Processing test dataset...'.format(test_dataset)) 
# config file for the dataset
# separates query image list from database image list, if revisited protocol used
cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))
print("1")

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()
print("2")
# # Load pretrained ResNet model
# model = models.resnet50(pretrained=True)
# model.eval()  # Set to evaluation mode

# Transformation pipeline for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features
def extract_features(image):
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()  # Convert to numpy array

# Placeholder for paths
data_root = 'path_to_data_root'
save_path = 'path_to_save_directory'
feature_filename_query = 'query_features.mat'
feature_filename_db = 'db_features.mat'


# Store features in lists
query_features = []
db_features = []

# Process query and database images
for i in np.arange(cfg['nq']):
    img_path = cfg['qim_fname'](cfg, i)
    img = pil_loader(img_path).crop(cfg['gnd'][i]['bbx'])
    features = extract_features(img)
    query_features.append(features)
    print('>> {}: Processing query image {}'.format(test_dataset, i+1))

for i in np.arange(cfg['n']):
    img_path = cfg['im_fname'](cfg, i)
    img = pil_loader(img_path)
    features = extract_features(img)
    db_features.append(features)
    print('>> {}: Processing database image {}'.format(test_dataset, i+1))

# Save features to .mat file
save_path = '/Users/liyunxiao/Desktop/revisitop/features'
feature_filename = 'resnet_feature.mat'
features = {'Q': query_features, 'X': db_features}

Q = features['Q']
X = features['X']

# 调整数组形状
Q_squeezed = np.squeeze(Q).T  # 从 (70, 1, 1000) 变为 (1000, 70)
X_squeezed = np.squeeze(X).T  # 从 (4993, 1, 1000) 变为 (1000, 4993)

# 保存调整后的特征到字典
new_features = {'Q': Q_squeezed, 'X': X_squeezed}

savemat(os.path.join(save_path, feature_filename), new_features)

print('Features saved to {}'.format(os.path.join(save_path, feature_filename)))


# # query images
# for i in np.arange(cfg['nq']):
#     img_path = cfg['qim_fname'](cfg, i)
#     img = pil_loader(img_path).crop(cfg['gnd'][i]['bbx'])
#     features = extract_features(img)
#     query_features.append(features)
#     print('>> {}: Processing query image {}'.format(test_dataset, i+1))
#     # qim = pil_loader(cfg['qim_fname'](cfg, i)).crop(cfg['gnd'][i]['bbx'])
#     # ##------------------------------------------------------
#     # ## Perform image processing here, eg, feature extraction
#     # ##------------------------------------------------------
#     # print('>> {}: Processing query image {}'.format(test_dataset, i+1))

# for i in np.arange(cfg['n']):
#     im = pil_loader(cfg['im_fname'](cfg, i))
#     ##------------------------------------------------------
#     ## Perform image processing here, eg, feature extraction
#     ##------------------------------------------------------
#     print('>> {}: Processing database image {}'.format(test_dataset, i+1))
