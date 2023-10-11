import torch
import config
import lightning_utils as li
from models.resnet_simclr import ResNetSimCLR


class XResNet50(li.BaseLightningModule):
    def __init__(self, loss_function, lr,
                 loss_one_hot=False,
                 use_softmax=False,
                 weight_decay=0.001,
                 scheduler_factor=0.1,
                 num_classes=config.NUM_CLASSES,
                 scheduler_patience=2,
                 scheduler_threshold=0.001):
        super().__init__(loss_function, lr,
                         loss_one_hot=loss_one_hot,
                         use_softmax=use_softmax,
                         weight_decay=weight_decay,
                         scheduler_factor=scheduler_factor,
                         num_classes=num_classes,
                         scheduler_patience=scheduler_patience,
                         scheduler_threshold=scheduler_threshold)
        self.resNet = ResNetSimCLR("xresnet1d50", leads=1, out_dim=config.NUM_CLASSES, hidden=True)
        self.batch_norm = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        # reshape the 1D to 2d
        x = self.batch_norm(x)
        x = self.resNet(x)
        # apply softmax
        return x


class XResNet101(li.BaseLightningModule):
    def __init__(self, loss_function, lr,
                 loss_one_hot=False,
                 use_softmax=False,
                 weight_decay=0.001,
                 scheduler_factor=0.1,
                 num_classes=config.NUM_CLASSES,
                 scheduler_patience=2,
                 scheduler_threshold=0.001):
        super().__init__(loss_function, lr,
                         loss_one_hot=loss_one_hot,
                         use_softmax=use_softmax,
                         weight_decay=weight_decay,
                         scheduler_factor=scheduler_factor,
                         num_classes=num_classes,
                         scheduler_patience=scheduler_patience,
                         scheduler_threshold=scheduler_threshold)
        self.resNet = ResNetSimCLR("xresnet1d101", leads=1, out_dim=config.NUM_CLASSES, hidden=True)
        self.batch_norm = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.batch_norm(x)
        # print(x.shape) # torch.Size([64, 1, 5000])
        x = self.resNet(x)
        # apply softmax
        return x


def get_sample_wise_preds(dataloader, nn):
    sample_to_trues = {}
    sample_to_pred_list = {}
    crop_pred = []
    crop_trues = []
    with torch.no_grad():
        for batch in dataloader:
            x, y_one_hot, ids = batch
            y_true_class = torch.argmax(y_one_hot.squeeze(), dim=1).long().detach()
            net_predictions = nn.forward(x).float().detach()
            crop_pred.append(net_predictions)
            crop_trues.append(y_true_class)
            for idx in range(net_predictions.shape[0]):
                crop_sample_id = ids[idx].item()
                sample_to_pred_list.setdefault(crop_sample_id, []).append(net_predictions[idx])

                if sample_to_trues.get(crop_sample_id) is None:
                    sample_to_trues[crop_sample_id] = y_true_class[idx]

    return sample_to_trues, sample_to_pred_list, torch.cat(crop_pred, dim=0), torch.cat(crop_trues, dim=0)
