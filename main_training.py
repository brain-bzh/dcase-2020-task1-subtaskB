import os
import sys
import time
import copy
import argparse

import tqdm
import random
import numpy as np

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
from torch.optim import lr_scheduler

from create_dataset import DATA_PATH, GRAPHS_PATH, MODEL_PATH, DEVICE
from create_dataset import create_dataset
from utils import load_model, save_model, plot_training, plot_cm, classif_report, rand_cut, EarlyStopping



print(f"Device: {DEVICE}")

### Training function ###

def train_model(model, hist, criterion, optimizer, dataloaders, dataset_sizes, 
                data_augment='', scheduler=None, num_epochs=25, patience_es=5):
    """
    Training function. 
    Return the trained model and a dictionary with the training info.
    """
    print("\n\n**TRAINING**\n")

    random_crop = True if data_augment == 'random_crop' else False
    cutmix = True if data_augment == 'cutmix' else False
    mixup = True if data_augment == 'mixup' else False

    print("RANDOM_CROP & Co:", random_crop)
    print("CUTMIX:", cutmix)
    print("MIXUP:", mixup)

    early_stopping = EarlyStopping(patience=patience_es, verbose=True, delta=0)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        lasttime = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            nb_batches = len(dataloaders[phase])

            # Iterate over data.
            pbar = tqdm.tqdm([i for i in range(nb_batches)])
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                pbar.update()
                pbar.set_description("Processing batch %s" % str(batch_idx+1))  
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    r = np.random.rand(1)
                    if phase == 'train' and (cutmix or mixup) and r <  0.8:
                        if cutmix:
                            # generate cutmixed sample
                            lam = np.random.beta(1,1)
                            target_a = labels
                            rand_index = torch.randperm(inputs.size()[0]).cuda()
                            target_b = labels[rand_index]
                            cut1, cut2 = rand_cut(inputs.size(), lam)

                            inputs[:, :, cut1:cut2] = inputs[rand_index, :, cut1:cut2]
                            outputs = model(inputs)
                            lam = 1 - ((cut2 - cut1)  / (inputs.size()[-1]))
                            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)

                        elif mixup:
                            # generate mixed sample
                            lam = np.random.beta(1,1)
                            target_a = labels
                            rand_index = torch.randperm(inputs.size()[0]).cuda()
                            target_b = labels[rand_index]

                            inputs[:, :, :] = inputs[:, :, :] * lam + inputs[rand_index, :, :] * (1. - lam)
                            outputs = model(inputs)
                            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)

                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            pbar.close()
            if phase == 'train' and scheduler != None and epoch != 0:
                scheduler.step(hist['val_loss'][-1])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(running_corrects.double(), dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))            

            hist[f'{phase}_loss'].append(epoch_loss)
            hist[f'{phase}_acc'].append(epoch_acc)

            # deep copy the model
            if phase == 'val':
                valid_loss = epoch_loss # Register validation loss for Early Stopping

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print("Epoch complete in {:.1f}s\n".format(time.time() - lasttime))

        # Check Early Stopping
        early_stopping(valid_loss, model)
    
        if early_stopping.early_stop:
            print("Early stopping")
            hist['best_val_acc'] = best_acc
            hist['epochs'] = np.arange(epoch + 1)
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    best_acc = round(float(best_acc), 4)
    print('Best val Acc: {:4f}'.format(best_acc))
    hist['best_val_acc'] = best_acc
    hist['epochs'] = np.arange(num_epochs)

    # load best model weights
    model.load_state_dict(best_model_wts)

    return (model, hist)


def test_model(model, hist, criterion, dataloaders, dataset_sizes, half=False):
    """
    Testing function. 
    Print the loss and accuracy after the inference on the testset.
    """
    print("\n\n**TESTING**\n")

    sincetime = time.time()

    phase = "test"
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    list_y_pred = []
    list_y_true = []
    list_probs = []

    nb_batches = len(dataloaders[phase])

    pbar = tqdm.tqdm([i for i in range(nb_batches)])

    # Iterate over data.
    for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
        pbar.update()
        pbar.set_description("Processing batch %s" % str(batch_idx+1))  
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # After Quantization
        if half:
            inputs = inputs.half()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = softmax(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        list_y_pred.append(int(preds.cpu()))
        list_y_true.append(int(labels.data.cpu()))
        list_probs.append(probs.cpu())

    pbar.close()

    test_loss = running_loss / dataset_sizes[phase]
    test_acc = running_corrects.double() / dataset_sizes[phase]
    test_acc = round(float(test_acc), 4)
    hist['test_acc'] = test_acc

    hist['y_pred'] = list_y_pred
    hist['probs'] = np.stack(list_probs).reshape(-1,3)
    hist['y_true'] = list_y_true

    print('\nTest stats -  Loss: {:.4f} Acc: {:.2f}%'.format(test_loss, test_acc*100))            

    print("Inference on Testset complete in {:.1f}s\n".format(time.time() - sincetime))

    return hist



if __name__ == "__main__":

    ### Main parameters ###
    parser = argparse.ArgumentParser()

    parser.add_argument('--saving', type=int, required=True)

    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True) 
    parser.add_argument('--scheduler', type=int, required=False, default=0) 

    parser.add_argument('--frac_data', type=float, required=False, default=1.)
    parser.add_argument('--da_mode', type=str, required=False, default="")

    args = parser.parse_args()


    # Path
    SAVING = args.saving                            # int 0: no, 1: yes

    # Model parameters
    MODEL_TYPE = args.model_type      # "ResNet18_1D" "VGG_1D"
    LR = args.lr                      # Learning Rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    SCHEDULER = args.scheduler        # int 0: no, 1: yes

    # Misc parameters
    FRAC_DATA = args.frac_data        # Take {FRAC_DATA}% of the dataset
    DATA_AUGMENT = args.da_mode  # 'cutmix' or 'mixup' or 'random_crop'
    RANDOM_STATE = 17
    random.seed(RANDOM_STATE)


    ### Data processing ###
    dataloaders_length = create_dataset(batch_size=BATCH_SIZE, frac_data=FRAC_DATA, 
                                        random_state=RANDOM_STATE, data_augment=DATA_AUGMENT)

    trainloader = torch.load(f'{DATA_PATH}train_data.pt')
    validationloader = torch.load(f'{DATA_PATH}val_data.pt')
    testloader = torch.load(f'{DATA_PATH}test_data.pt')

    dataloaders = {"train": trainloader,
                   "val": validationloader,
                   "test": testloader}

    dataset_sizes = {"train": dataloaders_length[0],
                     "val": dataloaders_length[1],
                     "test": dataloaders_length[2]}

    print(dataset_sizes)


    ### Load Model ###
    model = load_model(device=DEVICE, model_type=MODEL_TYPE)

    print(f"Model on device cuda: {next(model.parameters()).is_cuda}")

    ###  Define loss function and optimizer ### 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, threshold=1e-4) if SCHEDULER else None

    # Define dictionary for training info
    history_training = {'train_loss': [],
                        'val_loss': [],
                        'train_acc': [],
                        'val_acc': []}


    ### Training ###
    model, history_training = train_model(model=model, hist=history_training, criterion=criterion, 
                                          optimizer=optimizer, dataloaders=dataloaders, dataset_sizes=dataset_sizes, 
                                          data_augment=DATA_AUGMENT, scheduler=lr_sched, num_epochs=EPOCHS, patience_es= 15)


    ### Testing ###
    history_training = test_model(model=model, hist=history_training, criterion=criterion, 
                                  dataloaders=dataloaders, dataset_sizes=dataset_sizes)


    ### Save the model ###
    save_model(model=model, hist=history_training, 
               trained_models_path=MODEL_PATH, model_type=MODEL_TYPE, do_save=SAVING)


    ### Plotting the losses ###
    plot_training(hist=history_training, graphs_path=GRAPHS_PATH, 
                  model_type=MODEL_TYPE, do_save=SAVING)


    ### Plotting the CM ###
    plot_cm(hist=history_training, graphs_path=GRAPHS_PATH, 
                  model_type=MODEL_TYPE, do_save=SAVING)


    ### Give the classification report ###
    classif_report(hist=history_training)
