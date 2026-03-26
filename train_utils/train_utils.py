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
    net.load_state_dict(torch.load(save_path))
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
