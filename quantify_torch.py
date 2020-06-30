import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.optim import lr_scheduler

from create_dataset import DATA_PATH, MODEL_PATH, DEVICE
from create_dataset import create_dataset
from utils import load_model, get_model_type
from main_training import test_model



LIST_CLASSES = ['outdoor', 'indoor', 'transportation']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_filename', type=str, required=True)
    parser.add_argument('--frac_data', type=float, default=1.)

    args = parser.parse_args()

    TRAINED_MODEL = args.model_filename
    MODEL_TYPE = get_model_type(TRAINED_MODEL)
    BATCH_SIZE = 64

    # Misc parameters
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
    print(model)
    print(dataset_sizes)
    print(f"Model on device cuda: {next(model.parameters()).is_cuda}")


    ###  Define loss function ### 
    criterion = nn.CrossEntropyLoss()


    print("\nTesting before quantization to Half...\n")

    history_training = {}

    history_training = test_model(model=model, hist=history_training, criterion=criterion, 
                                  dataloaders=dataloaders, dataset_sizes=dataset_sizes, half=False)

    model = model.half()

    print("\nTesting after quantization to Half...\n")

    history_training = test_model(model=model, hist=history_training, criterion=criterion, 
                                  dataloaders=dataloaders, dataset_sizes=dataset_sizes, half=True)

    # Saving model
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    test_acc = history_training['test_acc']
    saved_model_path = f"{MODEL_PATH}{MODEL_TYPE}_{current_time}_quantified_testAcc={test_acc}.pth"
    torch.save(model, saved_model_path)
    print(f"Model saved at {saved_model_path}")
