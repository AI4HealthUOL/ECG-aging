import matplotlib.colors as mcolors
import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt

import config as config


def visualize_one_crop_raw(crop):
    y = crop.squeeze().numpy()
    datapoints = len(y)
    x_steps = np.arange(0, datapoints / 100, 0.01)

    plt.plot(x_steps, y, color="darkred")
    plt.xlabel("time in s")
    plt.ylabel("signal in mV")
    plt.show()


def visualize_one_crop(elem: torch.Tensor, size=3):
    datapoints = elem[0].shape[1]  # (1 per second so 1000Hz)
    label = elem[1].argmax(dim=0).item() + 1
    x_steps = np.arange(start=0, stop=size, step=size / datapoints)
    y = elem[0].numpy().transpose()
    plt.plot(x_steps, y, color="darkred")
    plt.xlabel("time in s")
    plt.ylabel("signal in mV")
    plt.show()
    print(f"predicted: {config.CLASSES[label]} sample is {elem[2].item()}")


def visualize_one_crop_grad(crop_x: torch.Tensor, grads: torch.Tensor, class_pred: int, text, normed=True, save=False,
                            file_name="file"):
    raw_signal = crop_x.detach().numpy()  # shape [1,1,size]
    w = grads.detach().numpy()  # shape [1,1,size]
    datapoints = raw_signal.shape[2]
    # normalize
    w_norm = 2 * (w - w.min()) / (w.max() - w.min()) - 1
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    if normed:
        w_norm = (w - w.min()) / (w.max() - w.min())

    cmap = plt.get_cmap("Reds")
    colors = cmap(w_norm[0, 0])
    x_ = np.arange(0, datapoints / 100, 0.01)

    plt.scatter(x_, raw_signal[0, 0], c=colors)
    plt.plot(x_, raw_signal[0, 0], c='black')
    plt.xlabel("time in s")
    plt.ylabel("signal in mV")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    if normed:
        sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm)
    if save:
        plt.savefig(file_name + ".png")
    plt.show()
    print(f"predicted: {config.CLASSES[class_pred + 1]} --> {text}")




def print_label_class_for_prediction(elem):
    label = elem.argmax(dim=0).item() + 1
    print(f"predicted: {config.CLASSES[label]}")


def visualize_as_spectogram(elem: torch.Tensor):
    image = F.to_pil_image(elem)
    plt.axis("off")
    plt.imshow(image)
    plt.show()
