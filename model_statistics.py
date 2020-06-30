import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.optim import lr_scheduler

from create_dataset import DATA_PATH, MODEL_PATH, CSV_PATH, DEVICE
from create_dataset import create_dataset
from utils import load_model, get_model_type, save_model, plot_training, plot_cm, classif_report, calculate_loss
from main_training import test_model
from pruning_torch import PrunerResnet, PrunerVGG



LIST_CLASSES = ['outdoor', 'indoor', 'transportation']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_filename', type=str, required=True)

    parser.add_argument('--frac_data', type=float, default=1.)
    parser.add_argument('--half', type=int, required=False, default=0)

    args = parser.parse_args()

    TRAINED_MODEL = args.model_filename
    MODEL_TYPE = get_model_type(TRAINED_MODEL)

    PRUNING_PERCENT = 0                         # Prune % of each layer
    NORME = 1                                   # Ln Norme
    DIMENSION = 0                               # Filters on axis=0

    BATCH_SIZE = 64

    # Misc parameters
    HALF = args.half                            # int 0: no, 1: yes (if the model has been halfed)
    FRAC_DATA = args.frac_data                  # Take {FRAC_DATA}% of the dataset
    DATA_AUGMENT = ''                           # 'cutmix' or 'random_crop' or 'mixup' or ''
    RANDOM_STATE = 17
    random.seed(RANDOM_STATE)


    ### Data processing ###
    dataloaders_length = create_dataset(batch_size=BATCH_SIZE, frac_data=FRAC_DATA, 
                                        random_state=RANDOM_STATE, data_augment=DATA_AUGMENT)

    testloader = torch.load(f'{DATA_PATH}test_data.pt')

    dataloaders = {"test": testloader}
    dataset_sizes = {"test": dataloaders_length[2]}

    model = load_model(device=DEVICE, saved_model_path=MODEL_PATH+TRAINED_MODEL)

    print(f"Model on device cuda: {next(model.parameters()).is_cuda}")


    ### Define loss function ### 
    criterion = nn.CrossEntropyLoss()

    ### Testing ### 
    history_training = {}

    history_training = test_model(model=model, hist=history_training, criterion=criterion, 
                                  dataloaders=dataloaders, dataset_sizes=dataset_sizes, half=HALF)

    # Give classification report
    classif_report(hist=history_training, list_names=LIST_CLASSES)

    # Give log losses
    calculate_loss(hist=history_training, list_names=LIST_CLASSES)

    ### Define Pruning class ###
    if MODEL_TYPE in ["ModelC", "ModelD"]:
        pruner_class = PrunerResnet(model=model, norme=NORME, amount=PRUNING_PERCENT, dimension=DIMENSION)

    elif MODEL_TYPE in ["ModelA", "ModelB"]:
        nb_layers = 4 if MODEL_TYPE == "ModelA" else 5 # "ModelA": VGG 4 layers | "ModelB": VGG 5 layers
        pruner_class = PrunerVGG(model=model, nb_layers=nb_layers, norme=NORME, amount=PRUNING_PERCENT, dimension=DIMENSION)

    else:
        raise ValueError('Pruning class not defined for this model.')


    print("\n== INFO ==\n")

    global_sparse, total_zero_weights, total_weights, df_stats = pruner_class.evaluate_sparse_all()

    print(df_stats)
    df_stats.to_csv(f'{CSV_PATH}{MODEL_TYPE}_layers.csv')

    # Calculate non zero parameters
    total_non_zero_with = 0 # With BatchNorm
    total_non_zero_without = 0 # Without BatchNorm
    for idx in range(len(df_stats)):
        if not df_stats['skip_count'][idx]:
            total_non_zero_without += df_stats['nzparams'][idx]
        total_non_zero_with += df_stats['nzparams'][idx]

    # (parameter values * 16 bits per parameter / 8 bits per byte / 1024 bytes per Byte)
    model_size_with = total_non_zero_with*16/8/1024 # model_size_with KB
    model_size_without = total_non_zero_without*16/8/1024 # model_size_without KB

    print("\n=============================================\n")

    print("Global sparsity: {:.2f}%\n".format(global_sparse))
    print("Total non-zero params (with batchNorm): {:d}\n".format(total_non_zero_with))
    print("Total non-zero params (without batchNorm): {:d}\n".format(total_non_zero_without))
    print("Total number of parameters: {:d}\n".format(total_weights))
    print("Model size (with batchNorm): {:.1f}KB\n".format(model_size_with))
    print("Model size (without batchNorm): {:.1f}KB\n".format(model_size_without))

    print("=============================================\n\n")
