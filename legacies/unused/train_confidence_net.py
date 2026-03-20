import copy

import seaborn as sns
from mmdet.apis import init_detector, inference_detector
import os
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
from tabulate import tabulate
from texttable import Texttable
import latextable
import seaborn as sns
import numpy as np
import h5py
import pandas as pd
import sklearn.metrics as sm
import math
import cv2
import copy
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
import sklearn.metrics as metrics
from skimage.transform import resize
from sklearn.metrics import precision_score, recall_score
from scipy.ndimage.filters import gaussian_filter
from shapely.geometry import Polygon
import torch
import torchvision.ops.boxes as bops
import slideio
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

# generate the simple linear regression net
class ConfidenceDataset(torch.utils.data.Dataset):
    '''
    Prepare the Confidence dataset for regression
    '''

    def __init__(self, X, y, scale_data=True, norm_mean=None, norm_std=None, mode='train', num_classes=2):
        self.num_classes = num_classes
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.scale_data = scale_data
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
            self.mode = mode
            # if scale_data:
            #     self.X = (self.X - torch.tensor(norm_mean).repeat(self.X.shape[0], 1))/torch.tensor(norm_std).repeat(self.X.shape[0], 1)
            #     self.X = torch.nan_to_num(self.X, nan=0.0, posinf=1.0)



    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        sample_features, sample_labels = self.X[i], self.y[i]

        # augment
        if self.mode == 'train':
            mu, sigma = 0, 0.15
            mu_out, sigma_out = 0, 0.1
            sample_features[:-self.num_classes] = sample_features[:-self.num_classes] + torch.from_numpy(np.random.normal(mu, sigma, sample_features.shape[0] - self.num_classes))
            sample_labels = sample_labels + torch.from_numpy(np.random.normal(mu_out, sigma_out, sample_labels.shape))

        if self.scale_data:
            sample_features[:-self.num_classes] = (sample_features[:-self.num_classes] - torch.tensor(self.norm_mean[:-self.num_classes]))/torch.tensor(self.norm_std[:-self.num_classes])
            sample_features = torch.nan_to_num(sample_features, nan=0.0, posinf=1.0, neginf=0.0)

        return sample_features, sample_labels


class MLP(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(401, 100),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            # nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            # nn.BatchNorm1d(50),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)


if __name__ == '__main__':

    # load the pd dataframe, set dirs and constants
    DATAFRAME_DIR = '/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_TMA.csv'
    RESULTS_SAVE_DIR = '/data/public/HULA/WSIs_renal_compartment_segmentations/'
    CLASSES = ["Glomerulus", "Arteriole", "Artery"]
    # set the val and test ids, strings for slide_id
    VAL_ID = 'WCM_'
    TEST_ID = 'Turin-'

    # set the model save path
    SAVE_PATH = '/data/public/HULA/WSIs_renal_compartment_segmentations/'

    # load the dataframe
    results_dataframe = pd.read_csv(DATAFRAME_DIR)

    # get the sub dfs
    val_id = 'WCM_'
    test_id = 'Turin-'
    val_df = results_dataframe.loc[
        (results_dataframe['slide_id'].str.contains(TEST_ID) == False) & results_dataframe['status'].str.contains('processed')]
    test_df = results_dataframe.loc[
        results_dataframe['slide_id'].str.contains(TEST_ID) & results_dataframe['status'].str.contains('processed')]
    test_train_df = results_dataframe.loc[
        (results_dataframe['slide_id'].str.contains(VAL_ID) == False) & results_dataframe['status'].str.contains('processed')]
    test_val_df = results_dataframe.loc[
        results_dataframe['slide_id'].str.contains(VAL_ID) & results_dataframe['status'].str.contains('processed')]
    val_df = val_df.replace('nan', np.NaN)
    test_df = test_df.replace('nan', np.NaN)
    test_val_df = test_val_df.replace('nan', np.NaN)
    test_train_df = test_train_df.replace('nan', np.NaN)
    df_list = [val_df, test_df]

    #######################################
    ########### TRAINING VAL TO TEST ######
    #######################################

    # get an np array with the features, targets for val
    data_np = np.empty(shape=(0, 1))
    norm_data_np = np.empty(shape=(0, 1))
    label_np = np.array([])
    slide_names = []
    for slide_id in val_df['slide_id']:
        slide_info = val_df.loc[val_df['slide_id'].str.contains(slide_id)]
        for class_id, class_names in enumerate(CLASSES):
            if (slide_info[class_names + '_f1_maxcut_rois'].iloc[0] != 'nan'):
                # get regression label
                label = np.array(slide_info[class_names + '_f1_maxcut_rois'].iloc[0])
                if np.isnan(label):
                    continue
                slide_names.append(slide_id + '_' + class_names)
                # get features
                raw_counts = np.array(slide_info[class_names + '_raw_count_histo_rois'].iloc[0].split(' ')).astype('float')
                freq_counts = np.array(slide_info[class_names + '_freq_count_histo_rois'].iloc[0].split(' ')).astype('float')
                # one hot encode class
                class_label = np.array(class_id)
                one_hot = np.zeros(len(CLASSES))
                one_hot[class_label] = 1
                # generate full feature vector
                sample_features = np.concatenate((raw_counts, freq_counts, one_hot))
                norm_sample_features = np.concatenate((raw_counts, freq_counts))
                # concatenate features and labels
                label_np = np.append(label_np, label)
                if data_np.shape == (0, 1):
                    data_np = np.append(data_np, sample_features[:, np.newaxis], axis=0)
                    norm_data_np = np.append(norm_data_np, norm_sample_features[:, np.newaxis], axis=0)
                else:
                    data_np = np.append(data_np, sample_features[:, np.newaxis], axis=1)
                    norm_data_np = np.append(norm_data_np, norm_sample_features[:, np.newaxis], axis=1)

    # get an np array with the features, targets for test
    test_data_np = np.empty(shape=(0, 1))
    test_norm_data_np = np.empty(shape=(0, 1))
    test_label_np = np.array([])
    test_slide_names = []
    for slide_id in test_df['slide_id']:
        slide_info = test_df.loc[test_df['slide_id'].str.contains(slide_id)]
        for class_id, class_names in enumerate(CLASSES):
            if (slide_info[class_names + '_f1_maxcut_rois'].iloc[0] != 'nan'):
                # get regression label
                label = np.array(slide_info[class_names + '_f1_maxcut_rois'].iloc[0])
                if np.isnan(label):
                    continue
                test_slide_names.append(slide_id + '_' + class_names)
                # get features
                raw_counts = np.array(slide_info[class_names + '_raw_count_histo_rois'].iloc[0].split(' ')).astype('float')
                freq_counts = np.array(slide_info[class_names + '_freq_count_histo_rois'].iloc[0].split(' ')).astype('float')
                # one hot encode class
                class_label = np.array(class_id)
                one_hot = np.zeros(len(CLASSES))
                one_hot[class_label] = 1
                # generate full feature vector
                sample_features = np.concatenate((raw_counts, freq_counts, one_hot))
                norm_sample_features = np.concatenate((raw_counts, freq_counts))
                # concatenate features and labels
                test_label_np = np.append(test_label_np, label)
                if test_data_np.shape == (0, 1):
                    test_data_np = np.append(test_data_np, sample_features[:, np.newaxis], axis=0)
                    test_norm_data_np = np.append(test_norm_data_np, norm_sample_features[:, np.newaxis], axis=0)
                else:
                    test_data_np = np.append(test_data_np, sample_features[:, np.newaxis], axis=1)
                    test_norm_data_np = np.append(test_norm_data_np, norm_sample_features[:, np.newaxis], axis=1)

    # get the train and val sets, this will change once we generate more data for test
    X_train = np.transpose(data_np)
    y_train = label_np
    X_val = np.transpose(test_data_np)
    y_val = test_label_np

    # get the norm values
    X_norm = np.transpose(norm_data_np)
    # norm_mean = [0.5] * X_norm.shape[1]
    # norm_std = [0.5] * X_norm.shape[1]
    norm_mean = list(np.mean(X_norm, axis=0))
    norm_std = list(np.std(X_norm, axis=0))
    norm_mean.extend([0 for _ in CLASSES])
    norm_std.extend([1 for _ in CLASSES])

    # save the values
    norm_np = np.concatenate((np.array(norm_mean)[:, np.newaxis], np.array(norm_std)[:, np.newaxis]), axis=1)
    np.save(SAVE_PATH + 'val_to_test_norm', norm_np)

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare Boston dataset
    train_dataset = ConfidenceDataset(X_train, y_train, scale_data=True, norm_mean=norm_mean, norm_std=norm_std, mode='train', num_classes=len(CLASSES))
    val_dataset = ConfidenceDataset(X_val, y_val, scale_data=True, norm_mean=norm_mean, norm_std=norm_std, mode='val', num_classes=len(CLASSES))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(y_train)//20, shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(y_val), shuffle=False, num_workers=1)

    # Initialize the MLP
    mlp = MLP()
    # mlp.cuda()

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=5e-5)

    # Run the training loop
    best_val_loss = 100.0
    best_R2 = -99999999999.9
    best_mse = 99999999999.9
    for epoch in range(0, 500):  # 500 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            # inputs = inputs.cuda()
            # targets = targets.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss))
            current_loss = 0.0

            # run val
            mlp.eval()
            with torch.no_grad():
                for i, data in enumerate(valloader, 0):

                    # Get and prepare inputs
                    inputs, targets = data
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], 1))

                    # targets = targets.cuda()
                    # inputs = inputs.cuda()

                    # perform forward pass
                    val_outputs = mlp(inputs)

                    # Compute loss
                    val_loss = loss_function(val_outputs, targets)

                    if val_loss.item() <= best_val_loss:
                        best_val_loss = val_loss.item()
                        print('Validation loss: {} (improved)'.format(val_loss))
                    else:
                        print('Validation loss: {}'.format(val_loss))

                    # val_outputs_np = np.array(val_outputs.cpu())
                    val_outputs_np = np.array(val_outputs)
                    # targets_np = np.array(targets.cpu())
                    targets_np = np.array(targets)
                    r2_val = sm.r2_score(targets_np, val_outputs_np)
                    mse_val = sm.mean_squared_error(targets_np, val_outputs_np)
                    vs_val = sm.explained_variance_score(targets_np, val_outputs_np)

                    if (r2_val >= best_R2) and (mse_val <= best_mse):
                        print('Val round performance: R2: {}. MSE: {} (complete improvement, saving)'.format(r2_val,
                                                                                                                     mse_val))
                        best_R2 = r2_val
                        best_mse = mse_val
                        torch.save(mlp, SAVE_PATH + 'val_to_test.pth')
                        save_pd = pd.DataFrame({'Names': test_slide_names, 'Pred.': val_outputs_np.flatten(), 'GT': targets_np.flatten(),
                                                'Diff.': (val_outputs_np.flatten() - targets_np.flatten())})
                        save_pd.to_csv(SAVE_PATH + 'val_to_test_results.csv', index=False)
                    else:
                        print('Val round performance: R2: {}. MSE: {}'.format(r2_val, mse_val))

            mlp.train()

    # Process is complete.
    print('Val to test model Training process has finished. R2: {}. MSE: {}'.format(best_R2, best_mse))

    #######################################
    ########### TRAINING VAL TO TEST ######
    #######################################

    # get an np array with the features, targets for val
    data_np = np.empty(shape=(0, 1))
    norm_data_np = np.empty(shape=(0, 1))
    label_np = np.array([])
    slide_names = []
    for slide_id in test_train_df['slide_id']:
        slide_info = test_train_df.loc[test_train_df['slide_id'].str.contains(slide_id)]
        for class_id, class_names in enumerate(CLASSES):
            if (slide_info[class_names + '_f1_maxcut_rois'].iloc[0] != 'nan'):
                # get regression label
                label = np.array(slide_info[class_names + '_f1_maxcut_rois'].iloc[0])
                if np.isnan(label):
                    continue
                slide_names.append(slide_id + '_' + class_names)
                # get features
                raw_counts = np.array(slide_info[class_names + '_raw_count_histo_rois'].iloc[0].split(' ')).astype('float')
                freq_counts = np.array(slide_info[class_names + '_freq_count_histo_rois'].iloc[0].split(' ')).astype('float')
                # one hot encode class
                class_label = np.array(class_id)
                one_hot = np.zeros(len(CLASSES))
                one_hot[class_label] = 1
                # generate full feature vector
                sample_features = np.concatenate((raw_counts, freq_counts, one_hot))
                norm_sample_features = np.concatenate((raw_counts, freq_counts))
                # concatenate features and labels
                label_np = np.append(label_np, label)
                if data_np.shape == (0, 1):
                    data_np = np.append(data_np, sample_features[:, np.newaxis], axis=0)
                    norm_data_np = np.append(norm_data_np, norm_sample_features[:, np.newaxis], axis=0)
                else:
                    data_np = np.append(data_np, sample_features[:, np.newaxis], axis=1)
                    norm_data_np = np.append(norm_data_np, norm_sample_features[:, np.newaxis], axis=1)

    # get an np array with the features, targets for test
    test_data_np = np.empty(shape=(0, 1))
    test_norm_data_np = np.empty(shape=(0, 1))
    test_label_np = np.array([])
    test_slide_names = []
    for slide_id in test_val_df['slide_id']:
        slide_info = test_val_df.loc[test_val_df['slide_id'].str.contains(slide_id)]
        for class_id, class_names in enumerate(CLASSES):
            if (slide_info[class_names + '_f1_maxcut_rois'].iloc[0] != 'nan'):
                # get regression label
                label = np.array(slide_info[class_names + '_f1_maxcut_rois'].iloc[0])
                if np.isnan(label):
                    continue
                test_slide_names.append(slide_id + '_' + class_names)
                # get features
                raw_counts = np.array(slide_info[class_names + '_raw_count_histo_rois'].iloc[0].split(' ')).astype('float')
                freq_counts = np.array(slide_info[class_names + '_freq_count_histo_rois'].iloc[0].split(' ')).astype('float')
                # one hot encode class
                class_label = np.array(class_id)
                one_hot = np.zeros(len(CLASSES))
                one_hot[class_label] = 1
                # generate full feature vector
                sample_features = np.concatenate((raw_counts, freq_counts, one_hot))
                norm_sample_features = np.concatenate((raw_counts, freq_counts))
                # concatenate features and labels
                test_label_np = np.append(test_label_np, label)
                if test_data_np.shape == (0, 1):
                    test_data_np = np.append(test_data_np, sample_features[:, np.newaxis], axis=0)
                    test_norm_data_np = np.append(test_norm_data_np, norm_sample_features[:, np.newaxis], axis=0)
                else:
                    test_data_np = np.append(test_data_np, sample_features[:, np.newaxis], axis=1)
                    test_norm_data_np = np.append(test_norm_data_np, norm_sample_features[:, np.newaxis], axis=1)

    # get the train and val sets, this will change once we generate more data for test
    X_train = np.transpose(data_np)
    y_train = label_np
    X_val = np.transpose(test_data_np)
    y_val = test_label_np

    # get the norm values
    X_norm = np.transpose(norm_data_np)
    norm_mean = [0.5] * X_norm.shape[1]
    norm_std = [0.5] * X_norm.shape[1]
    # norm_mean = list(np.mean(X_norm, axis=0))
    # norm_std = list(np.std(X_norm, axis=0))
    norm_mean.extend([0 for _ in CLASSES])
    norm_std.extend([1 for _ in CLASSES])

    # save the values
    norm_np = np.concatenate((np.array(norm_mean)[:, np.newaxis], np.array(norm_std)[:, np.newaxis]), axis=1)
    np.save(SAVE_PATH + 'test_to_val_norm', norm_np)

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare Boston dataset
    train_dataset = ConfidenceDataset(X_train, y_train, scale_data=True, norm_mean=norm_mean, norm_std=norm_std, mode='train', num_classes=len(CLASSES))
    val_dataset = ConfidenceDataset(X_val, y_val, scale_data=True, norm_mean=norm_mean, norm_std=norm_std, mode='val', num_classes=len(CLASSES))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(y_train)//20, shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(y_val), shuffle=False, num_workers=1)

    # Initialize the MLP
    mlp = MLP()
    # mlp.cuda()

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=5e-5)

    # Run the training loop
    best_val_loss = 100.0
    best_R2 = -99999999999.9
    best_mse = 99999999999.9
    for epoch in range(0, 500):  # 500 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            # inputs = inputs.cuda()
            # targets = targets.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss))
            current_loss = 0.0

            # run val
            mlp.eval()
            with torch.no_grad():
                for i, data in enumerate(valloader, 0):

                    # Get and prepare inputs
                    inputs, targets = data
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], 1))

                    # targets = targets.cuda()
                    # inputs = inputs.cuda()

                    # perform forward pass
                    val_outputs = mlp(inputs)

                    # Compute loss
                    val_loss = loss_function(val_outputs, targets)

                    if val_loss.item() <= best_val_loss:
                        best_val_loss = val_loss.item()
                        print('Validation loss: {} (improved)'.format(val_loss))
                    else:
                        print('Validation loss: {}'.format(val_loss))

                    # val_outputs_np = np.array(val_outputs.cpu())
                    val_outputs_np = np.array(val_outputs)
                    # targets_np = np.array(targets.cpu())
                    targets_np = np.array(targets)
                    r2_val = sm.r2_score(targets_np, val_outputs_np)
                    mse_val = sm.mean_squared_error(targets_np, val_outputs_np)
                    vs_val = sm.explained_variance_score(targets_np, val_outputs_np)

                    if (r2_val >= best_R2) and (mse_val <= best_mse):
                        print('Val round performance: R2: {}. MSE: {} (complete improvement, saving)'.format(r2_val,
                                                                                                                     mse_val))
                        best_R2 = r2_val
                        best_mse = mse_val
                        torch.save(mlp, SAVE_PATH + 'test_to_val.pth')
                        save_pd = pd.DataFrame({'Names': test_slide_names, 'Pred.': val_outputs_np.flatten(), 'GT': targets_np.flatten(),
                                                'Diff.': (val_outputs_np.flatten() - targets_np.flatten())})
                        save_pd.to_csv(SAVE_PATH + 'test_to_val_results.csv', index=False)
                    else:
                        print('Val round performance: R2: {}. MSE: {}'.format(r2_val, mse_val))

            mlp.train()

    # Process is complete.
    print('Test to val model Training process has finished. R2: {}. MSE: {}'.format(best_R2, best_mse))
