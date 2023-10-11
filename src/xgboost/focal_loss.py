import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossEntropyFocalLoss(nn.Module):
    """
    Focal CE loss for multiclass classification with integer labels
    Reference: https://github.com/artemmavrin/focal-loss/blob/7a1810a968051b6acfedf2052123eb76ba3128c4/src/focal_loss/_categorical_focal_loss.py#L162
    """
    def __init__(self, gamma=2, weight=[]):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight, dtype=np.float32)) if len(weight)>0 else None, reduction='none')

    def forward(self, preds, targs):
        probs = F.softmax(preds, dim=-1).squeeze(-1)
        probs = torch.gather(probs, -1, targs.unsqueeze(-1)).squeeze(-1)
        focal_modulation = torch.pow((1 - probs), self.gamma if type(self.gamma)==float else self.gamma.index_select(dim=0, index=preds.argmax(dim=-1)))
        # mean aggregation
        return (focal_modulation*self.ce(input=preds, target=targs)).mean()