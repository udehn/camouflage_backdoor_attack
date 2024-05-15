
from torchvision import datasets, models, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
import os

import numpy as np
from PIL import Image

import cv2

import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_cv2_transform(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(image)
    return image

def saveHistories(histories, saved_filename):
    pd.DataFrame(histories).to_csv(
        saved_filename,
        columns=['epoch',
        'acc_cl',
        'acc_bd',
        'loss_cl',
        'loss_bd'],
        index=False
    )

def updataHistories(histories, n_epoch,
                    test_orig_loader_accuracy,
                    test_trig_loader_accuracy,
                    test_orig_loader_loss,
                    test_trig_loader_loss
                    ):
    # histories['epoch'].append(len(histories['epoch']))
    histories['epoch'].append(n_epoch)
    histories['acc_cl'].append(test_orig_loader_accuracy)
    histories['acc_bd'].append(test_trig_loader_accuracy)
    histories['loss_cl'].append(test_orig_loader_loss)
    histories['loss_bd'].append(test_trig_loader_loss)


def evaluate(model, dataloader, loss_fn):
    losses = []

    num_correct = 0
    num_elements = 0

    for i, batch in enumerate(dataloader):
        X_batch, y_batch, _ = batch
        num_elements += len(y_batch)

        with torch.no_grad():
            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))
            losses.append(loss.item())

            y_pred = torch.argmax(logits, dim=1)

            num_correct += torch.sum(y_pred.cpu() == y_batch)

    accuracy = num_correct / num_elements

    return accuracy.numpy(), np.mean(losses)


# def train(model, loss_fn, optimizer, train_loader, test_orig_loader, test_trig_loader, histories, n_epoch=3):
def train(model, loss_fn, optimizer, train_loader, n_epoch=1):

    for epoch in range(n_epoch):
        # print("Epoch:", epoch + 1)
        model.train(True)

        running_losses = []
        running_accuracies = []
        for i, batch in enumerate(train_loader):
            X_batch, y_batch, _ = batch

            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))
            running_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            model_answers = torch.argmax(logits, dim=1)
            train_accuracy = torch.sum(y_batch == model_answers.cpu()) / len(y_batch)
            running_accuracies.append(train_accuracy)

            if (i + 1) % 100 == 0:
                print("Average train loss and accuracy in the last 100 iterations:",
                      np.mean(running_losses), np.mean(running_accuracies), end='\n')

        model.train(False)

        # test_orig_loader_accuracy, test_orig_loader_loss = evaluate(model, test_orig_loader, loss_fn=loss_fn)
        # print("Epoch {}/{}: test orig loss and accuracy:".format(epoch + 1, n_epoch, ),
        #       test_orig_loader_loss, test_orig_loader_accuracy, end='\n')
        #
        # test_trig_loader_accuracy, test_trig_loader_loss = evaluate(model, test_trig_loader, loss_fn=loss_fn)
        # print("Epoch {}/{}: test trig loss and accuracy:".format(epoch + 1, n_epoch, ),
        #       test_trig_loader_loss, test_trig_loader_accuracy, end='\n')
        #
        # histories['epoch'].append(len(histories['epoch']))
        # histories['acc_cl'].append(test_orig_loader_accuracy)
        # histories['acc_bd'].append(test_trig_loader_accuracy)
        # histories['loss_cl'].append(test_orig_loader_loss)
        # histories['loss_bd'].append(test_trig_loader_loss)

    return model


def create_model(model, num_freeze_layers, num_out_classes):
    model.fc = nn.Linear(512, num_out_classes)
    for i, layer in enumerate(model.children()):
        if i < num_freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False

    return model

