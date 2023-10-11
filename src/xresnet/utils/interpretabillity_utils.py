import neurokit2 as nk
import torch.nn
from captum.attr import Saliency

import visualization_utils as vu


def analyze_crop_saliency_grads(model, crop_x, text=""):
    saliency = Saliency(model)
    grads = saliency.attribute(crop_x)
    crop_out = model(crop_x)
    pred_idx = crop_out.argmax(dim=0).item()
    vu.visualize_one_crop_grad(crop_x, grads, pred_idx, text)


def get_crop_saliency_heartbeats(model, crop_x, true_class, _abs=True):
    try:
        crop_numpy = crop_x.squeeze().numpy()
        clean_crop = nk.ecg_clean(crop_numpy, sampling_rate=100)
        crop_peaks = nk.ecg_peaks(clean_crop, sampling_rate=100)
        peaks_raw = crop_peaks[1]["ECG_R_Peaks"]
        saliency = Saliency(model)
        grads = saliency.attribute(torch.stack((crop_x, crop_x)).squeeze(dim=1), target=true_class, abs=_abs)[
            0].unsqueeze(0)
        crop_out = model(crop_x).detach()
        grads = grads.detach()
        if _abs:
            grads = grads.abs()
        probs_out = torch.nn.functional.softmax(crop_out, dim=0)
        pred_out = torch.argmax(probs_out, dim=-1)
        # return two lists of org value and saliency values
        p, s = get_peaks(crop_x, grads, peaks_raw)
        return True, probs_out, pred_out, p, s
    except:
        return False, None, None, None, None


def get_crop_heart_beats(crop_x):
    try:
        crop_numpy = crop_x.squeeze().numpy()
        clean_crop = nk.ecg_clean(crop_numpy, sampling_rate=100)
        crop_peaks = nk.ecg_peaks(clean_crop, sampling_rate=100)
        peaks_raw = crop_peaks[1]["ECG_R_Peaks"]
        beats, s = get_peaks(crop_x, None, peaks_raw)
        return beats
    except:
        return []


def get_mean_heartbeat_for_crops(model, crops_in, true_class, only_peaks=False, _abs=True):
    grads = []
    probs = []
    preds = []
    peaks = []
    for crop in crops_in:
        if not only_peaks:
            ok, prob_c, pred_c, peaks_c, grads_c = get_crop_saliency_heartbeats(model, crop, true_class, _abs=_abs)
            if ok:
                probs.append(prob_c)
                preds.append(pred_c)
                grads.extend(grads_c)
                peaks.extend(peaks_c)
        else:
            peaks_c = get_crop_heart_beats(crop_x=crop)
            peaks.extend(peaks_c)
    print(f"found {len(peaks)} heartbeats in {len(crops_in)} crops")

    if not only_peaks:
        prob_mean = torch.stack(probs).mean(dim=0)
        pred_mean = prob_mean.argmax(dim=0).float().item()
        peaks_mean = torch.stack(peaks).mean(dim=0)
        grads_mean = torch.stack(grads).mean(dim=0)

        return pred_mean, peaks_mean, grads_mean
    else:
        peaks_mean = torch.stack(peaks).mean(dim=0)
        peaks_std = torch.stack(peaks).std(dim=0)[:, :, 40:70].mean().round(decimals=2).item()
        return peaks_mean, peaks_std


def get_mean_heartbeat_for_crops_best_ones(model, crops_in, class_trues, best_ones=100):
    grads = []
    probs = []
    preds = []
    peaks = []

    for crop in crops_in:
        ok, prob_c, pred_c, peaks_c, grads_c = get_crop_saliency_heartbeats(model, crop, class_trues)
        if ok:
            probs.append(prob_c)
            preds.append(pred_c)
            grads.append(grads_c)
            peaks.append(peaks_c)

    print(f"found {len(peaks)} heartbeats in {len(crops_in)} crops")

    indices = sorted(range(len(probs)), key=lambda i: torch.abs(probs[i][preds[i]]), reverse=True)[:best_ones]

    grads_b = []
    probs_b = []
    preds_b = []
    peaks_b = []

    for i in indices:
        probs_b.append(probs[i])
        preds_b.append(preds[i])
        grads_b.extend(grads[i])
        peaks_b.extend(peaks[i])

    prob_mean = torch.stack(probs_b).mean(dim=0)
    pred_mean = prob_mean.argmax(dim=0).float().item()
    peaks_mean = torch.stack(peaks_b).mean(dim=0)
    grads_mean = torch.stack(grads_b).mean(dim=0)

    return pred_mean, peaks_mean, grads_mean


def analyze_mean_crops_best_ones(model, crops_in, text, true_class, best_ones=100):
    pred_mean, peaks_mean, grads_mean = get_mean_heartbeat_for_crops_best_ones(model, crops_in, true_class, best_ones)
    vu.visualize_one_crop_grad(peaks_mean, grads_mean, pred_mean, text)


def analyze_mean_crops(model, crops_in, text, true_class, _abs=True, save=False, normed=False):
    pred_mean, peaks_mean, grads_mean = get_mean_heartbeat_for_crops(model, crops_in, true_class, _abs=_abs)
    vu.visualize_one_crop_grad(peaks_mean, grads_mean, pred_mean, text, normed=normed, save=save,
                               file_name=str(true_class) + "_grad")


def get_peaks(crop_x, grads, peaks_raw):
    time_left = 30
    time_right = 50
    peaks_org = []
    peaks_grads = []
    for peak_index in peaks_raw:
        left = peak_index - time_left
        if left < 0:
            # ignore this heartbeat
            break
        right = peak_index + time_right
        if right > 300:
            # ignore this heartbeat
            break
        peak = crop_x[:, :, left:right]
        if grads is not None:
            saliency = grads[:, :, left:right]
            peaks_grads.append(saliency)
        peaks_org.append(peak)

    return peaks_org, peaks_grads
