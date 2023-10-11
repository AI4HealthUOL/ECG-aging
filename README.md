# ECG-aging


1. Download the dataset from https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/ (last modified: 2021-06-14), and unpack it
2. change the paths in the config file, so that they suit your environment
3. install the common package via: pip install . (executed in the common folder) and also install all required dependencies
4. execute data_preparation_v12.ipynb.
5. to reproduce the xgboost results: execute xgboost_model_training_and_xai.ipynb
6. to reproduce the xresnet50 results execute the TRAIN, TEST and INTERPRET jupiter notebooks
7. to reproduce non-model results execute make_graph.ipynb and AnalyseHeartBeatsBetter.ipynb
