# Imports
import os
import time
from pathlib import Path

import config
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


# End Imports

def train_and_validate_model(model: LightningModule, train_dl, val_dl,
                             model_name: str,
                             epochs=50, fast_dev_run=False, enable_progress_bar=True, min_epochs=3):
    """
    :param val_dl:
    :param train_dl:
    :param epochs: the training epochs initial: 50
    :param lr: the learning rate initial 0.0002
    :param model: the lightning model that should be trained!
    :param model_name: the model name
    :return: a trained model
    """
    # now with pytorch lighning & gpu support
    torch.set_float32_matmul_precision("high")  # to make lightning happy
    logger = TensorBoardLogger(determine_logs_path(), name=model_name)
    pl.seed_everything(42, workers=True)  # for reproducibility
    precision = config.PRECISION
    start_time = time.time()
    # create model folder if it does not exist
    if not os.path.exists(determine_models_path()):
        Path(determine_models_path()).mkdir(parents=True, exist_ok=True)

    model_pathname = os.path.join(determine_models_path(), model_name)
    lr_logger = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        logger=logger,
        accelerator=('gpu' if torch.cuda.is_available() else 'cpu'),
        # using cpu only for debugging to ensure that the gpu is not the flaw
        min_epochs=min_epochs,
        max_epochs=epochs,
        precision=precision,
        callbacks=[EarlyStopping(monitor="val_sample_macro_auc", mode="max"), lr_logger],
        deterministic=False,
        fast_dev_run=fast_dev_run,
        enable_progress_bar=enable_progress_bar,
    )
    trainer.fit(model, train_dl, val_dl)
    print(f'Program runtime: {time.time() - start_time}')
    torch.save(model.state_dict(), model_pathname)


def determine_logs_path():
    return os.path.join(config.OUT_FOLDER, "logs")


def determine_models_path():
    return config.OUT_FOLDER
