import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from .losses import loss_factory
from tqdm import tqdm
import argparse
import csv
from pathlib import Path
# from smac import RunHistory
import json
import pdb


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Run deep learning experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    return parser.parse_args()

def main_training_loop(trainloader, net,
                       optimizer, scheduler, num_epochs, writer=None,
                       device=DEVICE, log_interval=5, config={},
                       save_path='models/trained_model.pth'):
    ''' Main (standard) training loop'''
    loss_fn = loss_factory[config['loss']['name']](**config['loss']['kwargs'])
    lowest_loss = torch.inf
    ds_len = len(trainloader)
    shape = None
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f'Epoch: {epoch+1}')
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            hsi_batch, rgb_batch, labels_batch = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(hsi_batch.to(torch.double).to(device), 
                          rgb_batch.to(torch.double).to(device))

            if shape is None:
                shape = outputs['preds'].shape

            # loss = loss_fn(outputs, labels_batch.reshape(shape).to(device))
            loss = loss_fn(outputs, labels_batch.to(device))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        ep_loss = running_loss/ds_len
        writer.add_scalar('Average Loss per Epoch', ep_loss, epoch)
        if epoch % log_interval == 0:
            print(f"loss:{ep_loss}")
        # save model if lowest loss
        if ep_loss < lowest_loss:
            lowest_loss = ep_loss
            print('saved')
            torch.save(net.state_dict(), save_path)
        scheduler.step(ep_loss)
    print('Done')
    return ep_loss


def test(testloader, net, save_path, num_classes, device=DEVICE):
    ''' Test via mIOU, GeneralizedDice metrics. TBA'''
    net.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    net.to(device)
    print("Num classes ", num_classes)
    miou = MeanIoU(num_classes=num_classes, per_class=True, input_format="index")
    gdice = GeneralizedDiceScore(num_classes=num_classes, include_background=False, input_format="index")
    predictions = []
    truth_labels = []
    with torch.no_grad():
        for data in testloader:
            hsi_batch, rgb_batch, labels_batch = data
            outputs = net(hsi_batch.to(torch.double).to(device), 
                          rgb_batch.to(torch.double).to(device))
            predictions.append(torch.argmax(outputs['preds'].cpu(), axis=1))
            truth_labels.append(torch.argmax(labels_batch, axis=1))
    
    preds = torch.cat(predictions, axis=0)
    gt_lbls = torch.cat(truth_labels, axis=0)

    if len(preds.shape) == 1:
        H, W = testloader.dataset.gt.shape[:-1]
        preds = preds.reshape(1, H, W)
        gt_lbls = gt_lbls.reshape(1, H, W)

    print("Going into miou")
    miou_score = miou(preds, gt_lbls).numpy()
    print("Going into gdice")
    gdice_score = gdice(preds, gt_lbls).numpy()
    return miou_score, gdice_score






import random
import matplotlib.pyplot as plt
def test_viz(testloader, net, save_path, num_classes, device=DEVICE, visualize=True):
    """Test via mIoU, GeneralizedDice metrics, with optional sample visualization."""
    net.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    net.to(device)
    net.eval()

    print("Num classes ", num_classes)

    miou = MeanIoU(num_classes=num_classes, per_class=True, input_format="index")
    gdice = GeneralizedDiceScore(
        num_classes=num_classes,
        include_background=False,
        input_format="index"
    )

    predictions = []
    truth_labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            hsi_batch, rgb_batch, labels_batch = data

            outputs = net(
                hsi_batch.to(torch.double).to(device),
                rgb_batch.to(torch.double).to(device)
            )

            # Predicted class indices: [B, H, W]
            batch_preds = torch.argmax(outputs['preds'].cpu(), dim=1)

            # Ground truth class indices: [B, H, W]
            batch_truth = torch.argmax(labels_batch, dim=1)

            predictions.append(batch_preds)
            truth_labels.append(batch_truth)

            # Visualize one random sample from this batch
            if visualize:
                sample_idx = random.randint(0, batch_preds.shape[0] - 1)

                pred_patch = batch_preds[sample_idx].numpy()   # shape [16, 16]
                true_patch = batch_truth[sample_idx].numpy()   # shape [16, 16]

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))

                im0 = axes[0].imshow(true_patch, vmin=0, vmax=num_classes - 1, interpolation='nearest')
                axes[0].set_title(f"Ground Truth (batch {batch_idx}, sample {sample_idx})")
                axes[0].axis("off")

                im1 = axes[1].imshow(pred_patch, vmin=0, vmax=num_classes - 1, interpolation='nearest')
                axes[1].set_title(f"Prediction (batch {batch_idx}, sample {sample_idx})")
                axes[1].axis("off")

                fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8, label="Class Index")
                plt.tight_layout()
                plt.show()

    preds = torch.cat(predictions, dim=0)
    gt_lbls = torch.cat(truth_labels, dim=0)

    if len(preds.shape) == 1:
        H, W = testloader.dataset.gt.shape[:-1]
        preds = preds.reshape(1, H, W)
        gt_lbls = gt_lbls.reshape(1, H, W)

    miou_score = miou(preds, gt_lbls).numpy()
    gdice_score = gdice(preds, gt_lbls).numpy()

    return miou_score, gdice_score
