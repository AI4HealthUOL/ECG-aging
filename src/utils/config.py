import os
from pathlib import Path
import numpy as np

# if you change any parameter in this config-file, you may need to run data_preparation_v12.ipynb again so that the changes arrive at the models

# you may need to adjust these two paths for you system!
SOURCE_FOLDER = os.path.join(Path(os.getcwd()).parent, "data") # path to folder where 'Autonomic-aging'-dataset is stored
TARGET_FOLDER = Path(os.getcwd()).parent # path where to store the models, the results etc.


# -------------no need to change anything below here if you just want to reproduce the results -------------#
DATASET_METADATA_FILE = os.path.join(SOURCE_FOLDER, 'subject-info.csv')
SPLIT_NAMES = ['test', 'val', 'train']  # process small datasets first so you don't have to wait long in case of errors
SPLIT_ARRAY = np.array([60, 20, 20]) # train-test-val-split ratio
ORIGINAL_SAMPLE_RATE = 1000 # in Hz
RESAMPLE_TO = 100 #in Hz, set to 0 to disable resampling
SAMPLING_RATE = RESAMPLE_TO if RESAMPLE_TO != 0 else ORIGINAL_SAMPLE_RATE

DATASET_OUT_NAME = os.path.join('paper_datasets', 'prepared_dataset_final_data_resampled_to_' + str(RESAMPLE_TO))
OUT_FOLDER = os.path.join(TARGET_FOLDER, DATASET_OUT_NAME)
EXTRACTED_ECGS_FOLDER = os.path.join(OUT_FOLDER, 'ECGs_to_analyse')
METADATA_CLEANED_FILE = os.path.join(OUT_FOLDER, 'metadata.csv')

LABEL_COLUMN_NAME = 'Age_group'
TENSOR_FILE_COLUMN_NAME = 'tensor_file'
NUM_WORKERS = 0 # for some reason dataloader-creation is a lot slower with NUM_WORKERS = os.cpu_count(). 0 has been the fastest in our experiments

# map age-bins to the same indices to merge them. First number is the age-bin from the original data,
# second number is the age-bin-number as which the age-bin will be treated
BIN_MERGE_DICT = {
    1: 1,#"18-19y",
    2: 2,#"20-24y",
    3: 3,#"25-29y",
    4: 4,#"30-34y",
    5: 5,#"35-39y",
    6: 6,#"40-44y",
    7: 7,#"45-49y",
    8: 8,#"50-54y",
    9: 9,#"55-59y",
    10: 10,#"60-64y",
    11: 11,#"65-69y",
    12: 12,#"70-74y",
    13: 13,#"75-79y",
    14: 14,#"80-84y",
    15: 15,#"85-92y"
}
# age-bin labels. If you merge age-bins, you may need to change their name here as well
CLASSES = {
    1: "18-19y",
    2: "20-24y",
    3: "25-29y",
    4: "30-34y",
    5: "35-39y",
    6: "40-44y",
    7: "45-49y",
    8: "50-54y",
    9: "55-59y",
    10: "60-64y",
    11: "65-69y",
    12: "70-74y",
    13: "75-79y",
    14: "80-84y",
    15: "85-92y"
}

#CLASS_NAMES = CLASSES.keys() # should be named class_indices and CLASSES.values() should be CLASS_NAMES
NUM_CLASSES = len(CLASSES)
AGE_GROUPS = np.array(list(CLASSES.keys()))
GROUP_NAMES = list(CLASSES.values())
PRECISION = "16-mixed"

# for xresnet only:
BATCH_SIZE = 1024
PIECE_LENGTH = 3 #split ecg singlas into 3s-crops
USE_OVERSAMPLING = False
PIECES_FOLDER = os.path.join(OUT_FOLDER, f'pieces_length_{PIECE_LENGTH}')
ONE_HOT_ENCODE_BINS = True # important when using a loss function like Cross-entropy encoding
choice = 'yes' if ONE_HOT_ENCODE_BINS else 'not'   
DATALOADER_FOLDER = os.path.join(OUT_FOLDER, f'dataloader_piece_length_{PIECE_LENGTH}_for_batchsize_{BATCH_SIZE}_{choice}_one_hot_encoded')
CLASS_NAMES = CLASSES.keys()