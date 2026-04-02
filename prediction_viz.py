import math
from typing import Callable 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


import torch

def make_torch_predict_fn(model, device="cuda"):
    model.eval()

    @torch.no_grad()
    def predict_fn(hsi_batch_np: np.ndarray, rgb_batch_np: np.ndarray) -> np.ndarray:
        # HSI: (B, 8, 8, 48)   -> (B, 48, 8, 8)
        # RGB: (B, 160, 160, 3)-> (B, 3, 160, 160)
        hsi_t = torch.from_numpy(hsi_batch_np).permute(0, 3, 1, 2).to(torch.double).to(device)
        rgb_t = torch.from_numpy(rgb_batch_np).permute(0, 3, 1, 2).to(torch.double).to(device)


        logits = model(hsi_t, rgb_t)['preds']  # expected (B, 21, 16, 16) or (B, 16, 16, 21)
        
        return logits.detach().cpu().numpy()

    return predict_fn


def _tile_starts(full_len: int, out_patch_len: int) -> list[int]:
    """
    Start indices on the OUTPUT/GT grid.
    Uses stride = out_patch_len, and adds a final overlapping tile if needed.
    """
    if full_len <= out_patch_len:
        return [0]

    starts = list(range(0, full_len - out_patch_len + 1, out_patch_len))
    last = full_len - out_patch_len
    if starts[-1] != last:
        starts.append(last)
    return starts


def reconstruct_full_prediction(
    hsi: np.ndarray,                     # (601, 2384, 48)
    rgb: np.ndarray,                     # (12020, 47680, 3)
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    hsi_patch_hw: tuple[int, int] = (8, 8),
    rgb_patch_hw: tuple[int, int] = (160, 160),
    out_patch_hw: tuple[int, int] = (16, 16),
    num_classes: int = 21,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Breaks the full HSI/RGB tensors into aligned patches, runs prediction,
    merges overlapping predictions by averaging logits, and returns:

      pred_labels: (1202, 4768, 1) int64
      pred_logits: (1202, 4768, 21) float32

    Assumptions enforced:
    - HSI, RGB, and output grids are perfectly aligned.
    - One output patch corresponds to one HSI patch and one RGB patch.
    """

    H_hsi, W_hsi, C_hsi = hsi.shape
    H_rgb, W_rgb, C_rgb = rgb.shape
    ph_hsi, pw_hsi = hsi_patch_hw
    ph_rgb, pw_rgb = rgb_patch_hw
    ph_out, pw_out = out_patch_hw

    # Infer full output size from the alignment ratios.
    # From the given data:
    #   601 -> 1202  (x2)
    #   2384 -> 4768 (x2)
    #   12020 -> 1202 (x1/10)
    #   47680 -> 4768 (x1/10)
    hsi_to_out_y = ph_out / ph_hsi
    hsi_to_out_x = pw_out / pw_hsi
    rgb_to_out_y = ph_rgb / ph_out
    rgb_to_out_x = pw_rgb / pw_out

    assert float(hsi_to_out_y).is_integer()
    assert float(hsi_to_out_x).is_integer()
    assert float(rgb_to_out_y).is_integer()
    assert float(rgb_to_out_x).is_integer()

    hsi_to_out_y = int(hsi_to_out_y)   # 2
    hsi_to_out_x = int(hsi_to_out_x)   # 2
    rgb_to_out_y = int(rgb_to_out_y)   # 10
    rgb_to_out_x = int(rgb_to_out_x)   # 10

    H_out_from_hsi = H_hsi * hsi_to_out_y
    W_out_from_hsi = W_hsi * hsi_to_out_x
    H_out_from_rgb = H_rgb // rgb_to_out_y
    W_out_from_rgb = W_rgb // rgb_to_out_x

    assert H_out_from_hsi == H_out_from_rgb, "HSI/RGB vertical alignment mismatch"
    assert W_out_from_hsi == W_out_from_rgb, "HSI/RGB horizontal alignment mismatch"

    H_out = H_out_from_hsi   # 1202
    W_out = W_out_from_hsi   # 4768

    # Sanity check that the patch-level mappings are consistent with the full tensors.
    assert ph_hsi * hsi_to_out_y == ph_out
    assert pw_hsi * hsi_to_out_x == pw_out
    assert ph_out * rgb_to_out_y == ph_rgb
    assert pw_out * rgb_to_out_x == pw_rgb

    y_starts_out = _tile_starts(H_out, ph_out)
    x_starts_out = _tile_starts(W_out, pw_out)

    # Accumulate logits and divide by counts at the end.
    logits_sum = np.zeros((H_out, W_out, num_classes), dtype=np.float32)
    counts = np.zeros((H_out, W_out, 1), dtype=np.float32)

    batch_hsi = []
    batch_rgb = []
    batch_meta = []

    def flush_batch():
        nonlocal batch_hsi, batch_rgb, batch_meta, logits_sum, counts
        if not batch_hsi:
            return

        hsi_batch = np.stack(batch_hsi, axis=0)   # (B, 8, 8, 48)
        rgb_batch = np.stack(batch_rgb, axis=0)   # (B, 160, 160, 3)

        # predict_fn must return either:
        #   (B, 16, 16, 21)  or  (B, 21, 16, 16)
        logits = predict_fn(hsi_batch, rgb_batch)
        

        if logits.ndim != 4:
            raise ValueError(f"predict_fn must return rank-4 logits, got shape {logits.shape}")

        if logits.shape[1:] == (ph_out, pw_out, num_classes):
            logits_hwc = logits
        elif logits.shape[1:] == (num_classes, ph_out, pw_out):
            logits_hwc = np.transpose(logits, (0, 2, 3, 1))
        else:
            raise ValueError(
                f"Unexpected logits shape {logits.shape}. "
                f"Expected (B,{ph_out},{pw_out},{num_classes}) or (B,{num_classes},{ph_out},{pw_out})."
            )

        for i, (y0, x0) in enumerate(batch_meta):
            logits_sum[y0:y0 + ph_out, x0:x0 + pw_out, :] += logits_hwc[i]
            counts[y0:y0 + ph_out, x0:x0 + pw_out, :] += 1.0

        batch_hsi = []
        batch_rgb = []
        batch_meta = []

    for y0_out in y_starts_out:
        for x0_out in x_starts_out:
            # Map output/GT coordinates to HSI and RGB coordinates.
            y0_hsi = y0_out // hsi_to_out_y
            x0_hsi = x0_out // hsi_to_out_x
            y0_rgb = y0_out * rgb_to_out_y
            x0_rgb = x0_out * rgb_to_out_x

            hsi_patch = hsi[y0_hsi:y0_hsi + ph_hsi, x0_hsi:x0_hsi + pw_hsi, :]
            rgb_patch = rgb[y0_rgb:y0_rgb + ph_rgb, x0_rgb:x0_rgb + pw_rgb, :]

            if hsi_patch.shape != (ph_hsi, pw_hsi, C_hsi):
                raise ValueError(f"Bad HSI patch at {(y0_hsi, x0_hsi)}: got {hsi_patch.shape}")
            if rgb_patch.shape != (ph_rgb, pw_rgb, C_rgb):
                raise ValueError(f"Bad RGB patch at {(y0_rgb, x0_rgb)}: got {rgb_patch.shape}")

            batch_hsi.append(hsi_patch)
            batch_rgb.append(rgb_patch)
            batch_meta.append((y0_out, x0_out))

            if len(batch_hsi) >= batch_size:
                flush_batch()

    flush_batch()

    pred_logits = logits_sum / np.clip(counts, 1e-8, None)
    pred_labels = np.argmax(pred_logits, axis=-1).astype(np.int64)[..., None]  # (1202, 4768, 1)

    return pred_labels, pred_logits


def plot_pred_vs_gt(pred_labels: np.ndarray, gt_labels: np.ndarray, num_classes: int = 21):
    """
    pred_labels: (1202, 4768, 1) or (1202, 4768)
    gt_labels:   (1202, 4768, 1) or (1202, 4768)
    """

    pred_2d = np.squeeze(pred_labels)
    gt_2d = np.squeeze(gt_labels)

    if pred_2d.shape != gt_2d.shape:
        raise ValueError(f"Shape mismatch: pred {pred_2d.shape}, gt {gt_2d.shape}")

    # 21 distinct colors
    colors = list(plt.cm.tab20.colors) + [plt.cm.tab20b.colors[0]]
    cmap = ListedColormap(colors[:num_classes])

    fig, axes = plt.subplots(1, 2, figsize=(24, 8), constrained_layout=True)

    im0 = axes[0].imshow(pred_2d, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation="nearest")
    axes[0].set_title("Predicted labels")
    axes[0].axis("off")

    im1 = axes[1].imshow(gt_2d, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation="nearest")
    axes[1].set_title("Ground truth labels")
    axes[1].axis("off")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_ticks(np.arange(num_classes))
    cbar.set_label("Class index")

    plt.show()



def full_viz(model, hsi_full, rgb_full, gt_full):
    predict_fn = make_torch_predict_fn(model, device="cpu")
    
    pred_labels, pred_logits = reconstruct_full_prediction(
        hsi_full,
        rgb_full,
        predict_fn,
        num_classes=21,
        batch_size=32,
    )
    
    print(pred_labels.shape)   # (1202, 4768, 1)
    
    plot_pred_vs_gt(pred_labels, gt_full, num_classes=21)
