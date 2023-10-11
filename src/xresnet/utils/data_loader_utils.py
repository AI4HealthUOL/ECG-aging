# IMPORTS

import os
import platform
from pathlib import Path

import numpy as np
import torch

import config


# import scikitplot as skplt
# END IMPORTS


def determine_data_sets_path():
    return config.DATALOADER_FOLDER


def load_data_loaders(dataset_to_use: str, do_spectrograms=False, n_fft=200, one_hot_encoded_bins=True, batch_size=64,
                      piece_length=3):
    """

    :param one_hot_encoded_bins:
    :rtype: Tuple
    :param dataset_to_use: the name of the dataset to use
    :param batch_size: the batch size
    :param piece_length: the crop length
    :return: tuple of shape (ds_train, ds_val, ds_test, classWeights)
    """
    choice = 'not'
    if one_hot_encoded_bins:
        choice = 'yes'

    out_folder = os.path.join(determine_data_sets_path(), dataset_to_use)
    dataloader_folder = os.path.join(out_folder,
                                     f'dataloader_piece_length_{piece_length}_for_batchsize_{batch_size}_{choice}_one_hot_encoded')

    dataloader_set = []
    if os.path.exists(dataloader_folder):
        for set_name in config.SPLIT_NAMES:
            if not do_spectrograms:
                dataloader_set.append(torch.load(os.path.join(dataloader_folder, set_name + '_DL.pt')))
            else:
                dataloader_set.append(
                    torch.load(os.path.join(dataloader_folder, set_name + '_spectrograms_fft' + str(n_fft) + '_DL.pt')))
    else:
        raise ValueError(f'Folder for dataloaders {dataloader_folder} does not exist!')

    test_dl = dataloader_set[0]
    val_dl = dataloader_set[1]
    train_dl = dataloader_set[2]

    # check if shape is correct
    sample_batch = torch.utils.data.Subset(val_dl, [0]).dataset
    sample = sample_batch.dataset[0]
    x_test = sample[0]
    y_test = sample[1]
    id_test = sample[2]
    x_shape = x_test.shape

    if sample_batch.batch_size != batch_size:
        raise ValueError(
            f'ERROR! Batch size of dataloder is {x_shape[0]} but program configured for batchsize {batch_size}')

    y_train = np.array([i[1].numpy() for i in train_dl.dataset])
    class_weight_list = []
    for val in config.CLASS_NAMES:
        num_occurences_class = 0
        if not one_hot_encoded_bins:
            num_occurences_class = np.count_nonzero(y_train == val)
        else:
            for arr in y_train:
                if arr[val - 1] == 1:
                    num_occurences_class += 1

        if num_occurences_class == 0:
            num_occurences_class = 1  # to avoid inf-weights

        class_weight_list.append(num_occurences_class)

    class_weight_array = np.array(class_weight_list) / len(y_train)
    # contains relative percentages of the labels in the train set
    class_weights = torch.tensor(1 / class_weight_array).float()  # value class more if it is underrepresented

    input_datasize_flattend = len(torch.flatten(train_dl.dataset[0][0]))
    print("Loaded Dataloaders ")
    print(f"class_weights: {class_weights}")
    print(f"input_datasize_flattend: {input_datasize_flattend}")
    print("---------------------------------------------------")
    print(f"Train samples: {len(get_samples_and_count(train_dl))}")
    print(f"val samples: {len(get_samples_and_count(val_dl))}")
    print(f"test samples: {len(get_samples_and_count(test_dl))}")
    return train_dl, val_dl, test_dl, class_weights


def get_samples_and_count(dataloader):
    ids = {}
    for x, y, i in dataloader:
        for _id in i:
            _id = _id.item()
            ids[_id] = 1 if not ids.__contains__(_id) else ids[_id] + 1
    return ids


def get_samples_for_age_bin_and_patients(dataloader):
    sample_id_to_crop_list = {}
    sample_id_to_age_bin = {}
    age_bin_to_crop_list = {}
    for batch in dataloader:
        x_s = batch[0]
        y_s = batch[1]
        id_s = batch[2]
        for x, y, sample_id in zip(x_s, y_s, id_s):
            age_bin = y.detach().argmax(dim=0).item() + 1
            crop = x.unsqueeze(dim=0)
            sample_id = sample_id.detach().item()
            sample_crops = sample_id_to_crop_list.setdefault(sample_id, [])
            sample_crops.append(crop)
            sample_id_to_crop_list[sample_id] = sample_crops
            sample_id_to_age_bin[sample_id] = age_bin
            age_bin_crops = age_bin_to_crop_list.setdefault(age_bin, [])
            age_bin_crops.append(crop)
            age_bin_to_crop_list[age_bin] = age_bin_crops

    return sample_id_to_crop_list, age_bin_to_crop_list, sample_id_to_age_bin
