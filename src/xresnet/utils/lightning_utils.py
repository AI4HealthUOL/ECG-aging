import config
import numpy as np
import pytorch_lightning as pl
import torch.nn
import torchmetrics


class BaseLightningModule(pl.LightningModule):
    def __init__(self, loss_function, lr,
                 loss_one_hot=False,
                 use_softmax=False,
                 weight_decay=0.001,
                 scheduler_factor=0.1,
                 num_classes=config.NUM_CLASSES,
                 scheduler_patience=2,
                 scheduler_threshold=0.001):
        super().__init__()
        self.lr = lr
        self.use_softmax = use_softmax
        self.weight_decay = weight_decay
        self.factor = scheduler_factor
        self.patience = scheduler_patience
        self.threshold = scheduler_threshold
        self.loss_fn = loss_function
        self.loss_one_hot = loss_one_hot
        self.softmax = torch.nn.Softmax(dim=-1)

        self.sample_id_to_predictions_list_dict = {}  # dict that stores for each samp a list of its results
        self.sample_id_to_true_class_dict = {}
        self.sample_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.sample_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
        self.sample_f1_score_class = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="none")
        self.sample_auc_class = torchmetrics.AUROC(task="multiclass", num_classes=num_classes, average="none")
        self.sample_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

        self.crop_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.crop_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

    def _common_step(self, batch):
        x, y, crops_sample_ids = batch
        y_true_class = torch.argmax(y.squeeze(), dim=1).long()  # undo one-hot-encoding
        net_predictions = self.forward(x).float()  # append softmax so we have probs
        if self.use_softmax:
            net_predictions = self.softmax(net_predictions).float()

        for idx in range(net_predictions.shape[0]):  # shape of predictions: [batch_size,num_classes]?
            # for each crop in batch: append its prediction to the sample-dict

            # get sample-id of this crop
            crop_sample_id = crops_sample_ids[idx].item()  # try to avoid .item(), it yields slow code
            # append it to the overview-dict
            self.sample_id_to_predictions_list_dict.setdefault(crop_sample_id, []).append(net_predictions[idx])

            # is this sample-id already part of the dict?
            if self.sample_id_to_true_class_dict.get(crop_sample_id) is None:
                self.sample_id_to_true_class_dict[crop_sample_id] = y_true_class[idx]

        # metrics update
        self.crop_f1_score.update(net_predictions, y_true_class)
        self.crop_auc.update(net_predictions, y_true_class)
        loss = self.loss_fn(net_predictions, y) if self.loss_one_hot else self.loss_fn(net_predictions, y_true_class)

        if np.isnan(loss.detach().cpu()):  # detach, otherwise runtime is increased
            print(f'problem: loss is NaN! input ok ? {not x.isnan().any()} y ok? {not y.isnan().any()}')
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, rank_zero_only=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, rank_zero_only=True)
        return loss

    def calc_sample_metrics_epoch(self, log_prefix, do_print=False):
        sample_avg_prediction_list = []
        sample_true_class_list = []
        for _id, outs in self.sample_id_to_predictions_list_dict.items():
            # build the mean
            avg_prediction_for_sample = torch.mean(torch.stack(outs), dim=0)
            sample_avg_prediction_list.append(avg_prediction_for_sample)
            sample_true_class_list.append(self.sample_id_to_true_class_dict[_id])

        samples_prediction_tensor = torch.stack(sample_avg_prediction_list)
        samples_true_class_tensor = torch.stack(sample_true_class_list)

        # print(len(sample_avg_prediction_list))

        # calc metrics
        self.sample_auc.update(samples_prediction_tensor, samples_true_class_tensor)
        self.sample_f1_score.update(samples_prediction_tensor, samples_true_class_tensor)
        self.sample_auc_class.update(samples_prediction_tensor, samples_true_class_tensor)
        self.sample_f1_score_class.update(samples_prediction_tensor, samples_true_class_tensor)
        self.sample_confusion_matrix.update(samples_prediction_tensor, samples_true_class_tensor)

        macro_auc = self.sample_auc.compute()
        macro_f1 = self.sample_f1_score.compute()
        auc_class = self.sample_auc_class.compute()
        f1_class = self.sample_f1_score_class.compute()
        confmat = self.sample_confusion_matrix.compute()

        # log
        self.log(log_prefix + "_sample_macro_auc", macro_auc, prog_bar=True, on_step=False, on_epoch=True,
                 rank_zero_only=True)
        self.log(log_prefix + "_sample_macro_f1", macro_f1, prog_bar=True, on_step=False, on_epoch=True,
                 rank_zero_only=True)

        if do_print:
            print(f"Sample {log_prefix} macro-AUC of {macro_auc} in epoch {self.current_epoch}")
            print(f"Sample {log_prefix} macro-F1 of {macro_f1} in epoch {self.current_epoch}")
            print(f"Sample {log_prefix} conf mat: \n{confmat}\n in epoch {self.current_epoch}")

        # get scores per class
        for _id, _class_name in config.CLASSES.items():
            true_id = _id - 1  # since we start with 1 in the config
            _auc_class = auc_class[true_id]
            _f1_class = f1_class[true_id]
            self.log(log_prefix + "_sample_auc_class_" + str(_id), _auc_class,
                     prog_bar=False,
                     on_step=False,
                     on_epoch=True,
                     rank_zero_only=True)
            self.log(log_prefix + "_sample_f1_class_" + str(_id), _f1_class,
                     prog_bar=False,
                     on_step=False,
                     on_epoch=True,
                     rank_zero_only=True)

    def reset_sample_metrics(self):
        print(f"There had been {len(self.sample_id_to_true_class_dict)} samples")
        self.sample_id_to_predictions_list_dict = {}
        self.sample_id_to_true_class_dict = {}
        self.sample_auc.reset()
        self.sample_f1_score.reset()
        self.sample_auc_class.reset()
        self.sample_f1_score_class.reset()
        self.sample_confusion_matrix.reset()

    def on_train_epoch_start(self) -> None:
        self.reset_sample_metrics()
        self.reset_crop_metrics()

    def on_validation_epoch_start(self) -> None:
        self.reset_sample_metrics()
        self.reset_crop_metrics()

    def on_test_epoch_start(self) -> None:
        self.reset_sample_metrics()
        self.reset_crop_metrics()

    def calc_crop_metrics_epoch(self, log_prefix, do_print=False):
        macro_f1 = self.crop_f1_score.compute()
        macro_auc = self.crop_auc.compute()

        self.log(log_prefix + "_crop_macro_f1", macro_f1, prog_bar=True, on_step=False, on_epoch=True,
                 rank_zero_only=True)
        self.log(log_prefix + "_crop_macro_auc", macro_auc, prog_bar=True, on_step=False, on_epoch=True,
                 rank_zero_only=True)

        if do_print:
            print(f"Crop {log_prefix} macro-AUC of {macro_auc} in epoch {self.current_epoch}")
            print(f"Crop {log_prefix} macro-F1 of {macro_f1} in epoch {self.current_epoch}")

    def reset_crop_metrics(self):
        self.crop_auc.reset()
        self.crop_f1_score.reset()

    def on_training_epoch_end(self) -> None:
        self.calc_crop_metrics_epoch("train")
        self.calc_sample_metrics_epoch("train")

    def on_validation_epoch_end(self) -> None:
        self.calc_crop_metrics_epoch("val", True)
        self.calc_sample_metrics_epoch("val", True)

    def on_test_epoch_end(self) -> None:
        self.calc_crop_metrics_epoch("test", True)
        self.calc_sample_metrics_epoch("test", True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.factor,
                                                               patience=self.patience,
                                                               threshold=self.threshold)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }


def ensure_one_hot_labels(y):
    if y.shape[1] != config.NUM_CLASSES:
        label_pos_real = y - 1  # remove the extra label so we start with age bin 0
        label_pos_real = label_pos_real.long()
        label_pos_real = label_pos_real.squeeze()  # reduce the dimension
        y_hot_k = torch.nn.functional.one_hot(label_pos_real, config.NUM_CLASSES).float()
        return y_hot_k
    return y
