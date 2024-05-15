from torchvision import datasets, models, transforms

from torch.utils.data import random_split, DataLoader, Dataset
import torch
from torch import nn
from tqdm import tqdm

import joblib
from torch.optim.lr_scheduler import StepLR
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches

from PIL import Image

import os
from config import get_arguments
from trainModel import create_model, train, evaluate, updataHistories, saveHistories

from data_loader import get_backdoor_loader, get_test_loader

def totrain(opt):
    # Load models
    print('----------- Model Initialization --------------')

    train_model = create_model(models.resnet18(pretrained=True), 6, 10)
    device = torch.device('cuda')
    train_model.to(device)

    print('Finish Loading Models...')

    # initialize optimizer
    learning_rate = [0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]
    optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate[0])

    # define loss functions
    loss_fn = torch.nn.CrossEntropyLoss()
    # if opt.cuda:
    #     loss_fn = nn.CrossEntropyLoss().to(opt.device)
    # else:
    #     loss_fn = nn.CrossEntropyLoss()

    print('----------- Data Initialization --------------')

    _, train_data_bad_loader = get_backdoor_loader(opt)
    test_data_clean_loader, test_data_bad_loader = get_test_loader(opt)

    histories = {
        'epoch': [],
        'acc_cl': [],
        'acc_bd': [],
        'loss_cl': [],
        'loss_bd': []
    }

    print('----------- Training Backdoored Model --------------')
    for epoch in range(0, 30):
        print("Epoch: ", epoch)

        train_model = train(train_model, loss_fn, optimizer,
                            train_data_bad_loader,
                            n_epoch=1)

        optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate[int(epoch/6)])

        test_orig_loader_accuracy, test_orig_loader_loss = evaluate(train_model, test_data_clean_loader, loss_fn=loss_fn)
        test_trig_loader_accuracy, test_trig_loader_loss = evaluate(train_model, test_data_bad_loader, loss_fn=loss_fn)
        print("Epoch {}/{}: test orig loss and accuracy:".format(epoch + 1, 30, ),
              test_orig_loader_loss, test_orig_loader_accuracy, end='\n')

        print("Epoch {}/{}: test trig loss and accuracy:".format(epoch + 1, 30, ),
              test_trig_loader_loss, test_trig_loader_accuracy, end='\n')

        updataHistories(histories, epoch,
                        test_orig_loader_accuracy,
                        test_trig_loader_accuracy,
                        test_orig_loader_loss,
                        test_trig_loader_loss
                        )

    print('---------------- Finish Training -------------------')
    saved_filename = './logs/train_histories.csv'
    saveHistories(histories, saved_filename)
    torch.save(train_model, './saved_models/resnet18-whole_model-attack_demo.pth')

def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    totrain(opt)


if __name__ == '__main__':
    main()
