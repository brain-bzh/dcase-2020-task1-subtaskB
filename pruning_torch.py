import random
import argparse
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.optim import lr_scheduler

from create_dataset import DATA_PATH, MODEL_PATH, DEVICE
from create_dataset import create_dataset
from utils import load_model, get_model_type
from main_training import train_model, test_model



def percent_sparse(layer, do_print=False):
        n_weights = int(layer.weight.nelement())
        nzero_weights = int(torch.sum(layer.weight == 0))

        if do_print:
            percent = 100. * nzero_weights / n_weights
            print("Sparsity in {}: {:.2f}%".format(layer, percent))

        return (nzero_weights, n_weights)

def create_df_stats(liste, list_names):
    params_all = []
    print(list_names)
    for curname, curlayer in zip(list_names, liste):
        zero_weights = int(curlayer[0])
        weights = int(curlayer[1])
        sparsity = 100 * round(zero_weights / weights, 2)
        skip_count = True if 'bn' in curname else False
        params_all.append(dict(name=curname, params=weights, 
                               nzparams=weights-zero_weights, 
                               sparsity=sparsity, skip_count=skip_count))
    df_stats = pd.DataFrame(params_all)

    return df_stats

def give_summary_sparse(liste, list_names):
    total_zero_weights = int(sum([nzero_weights for nzero_weights, n_weights in liste]))
    total_weights = int(sum([n_weights for nzero_weights, n_weights in liste]))
    global_sparse = round(100 * total_zero_weights / total_weights, 2)

    print("\nGlobal sparsity: {:.2f}%".format(global_sparse))
    print("Total number of non-zero parameters: {:d}".format(total_weights-total_zero_weights))
    print("Total number of parameters: {:d}\n".format(total_weights))


    df_stats = create_df_stats(liste, list_names)

    return global_sparse, total_zero_weights, total_weights, df_stats


class PrunerResnet:
    def __init__(self, model, norme, amount, dimension):
        self.model = model
        self.norme = norme
        self.amount = amount
        self.dim = dimension
        self.list_sparse = []
        self.list_layers = [layername 
                            for layername, _ in self.model.named_modules()
                            if 'conv' in layername
                            or 'bn' in layername
                            or 'shortcut.0' in layername
                            or 'shortcut.1' in layername
                            or 'linear' in layername]

    def prune_block(self, sub_layer):
        for block_num, block in enumerate(sub_layer):
            prune.ln_structured(module=block.conv1, name='weight', amount=self.amount, n=self.norme, dim=self.dim)
            prune.l1_unstructured(module=block.bn1, name='weight', amount=self.amount)
            prune.ln_structured(module=block.conv2, name='weight', amount=self.amount, n=self.norme, dim=self.dim)
            prune.l1_unstructured(module=block.bn2, name='weight', amount=self.amount)
            for short_layer in block.shortcut:
                if isinstance(short_layer, torch.nn.modules.conv.Conv1d):
                    prune.ln_structured(module=short_layer, name='weight', amount=self.amount, n=self.norme, dim=self.dim)
                elif isinstance(short_layer, torch.nn.modules.batchnorm.BatchNorm1d):
                    prune.l1_unstructured(module=short_layer, name='weight', amount=self.amount)

    def prune_all(self):
        prune.ln_structured(module=self.model.conv1, name='weight', amount=self.amount, n=self.norme, dim=self.dim)
        prune.l1_unstructured(module=self.model.bn1, name='weight', amount=self.amount)
        self.prune_block(self.model.layer1)
        self.prune_block(self.model.layer2)
        self.prune_block(self.model.layer3)
        prune.ln_structured(module=self.model.linear, name='weight', amount=self.amount, n=self.norme, dim=self.dim)

    def evaluate_sparse_block(self, sub_layer):
        for block_num, block in enumerate(sub_layer):
            self.list_sparse.append(percent_sparse(block.conv1))
            self.list_sparse.append(percent_sparse(block.bn1))
            self.list_sparse.append(percent_sparse(block.conv2))
            self.list_sparse.append(percent_sparse(block.bn2))
            for short_layer in block.shortcut:
                if isinstance(short_layer, torch.nn.modules.conv.Conv1d):
                    self.list_sparse.append(percent_sparse(short_layer))
                elif isinstance(short_layer, torch.nn.modules.batchnorm.BatchNorm1d):
                    self.list_sparse.append(percent_sparse(short_layer))

    def evaluate_sparse_all(self):
        self.list_sparse.append(percent_sparse(self.model.conv1))
        self.list_sparse.append(percent_sparse(self.model.bn1))
        self.evaluate_sparse_block(model.layer1)
        self.evaluate_sparse_block(model.layer2)
        self.evaluate_sparse_block(model.layer3)
        self.list_sparse.append(percent_sparse(self.model.linear))

        global_sparse, total_zero_weights, total_weights, df_stats = give_summary_sparse(liste=self.list_sparse, 
                                                                                         list_names=self.list_layers)

        # Reset list for other pruning iterations
        self.list_sparse = []

        return global_sparse, total_zero_weights, total_weights, df_stats


class PrunerVGG:
    def __init__(self, model, nb_layers, norme, amount, dimension):
        self.model = model
        self.nb_layers = nb_layers
        self.norme = norme
        self.amount = amount
        self.dim = dimension
        self.list_sparse = []
        self.list_layers = [layername 
                            for layername, _ in self.model.named_modules()
                            if 'conv' in layername
                            or 'bn' in layername
                            or 'fc1' in layername]

    def prune_all(self):
        for layer_idx in range(self.nb_layers):
            conv = eval(f"self.model.conv{layer_idx+1}")
            bn = eval(f"self.model.bn{layer_idx+1}")
            prune.ln_structured(module=conv, name='weight', amount=self.amount, n=self.norme, dim=self.dim)
            prune.l1_unstructured(module=bn, name='weight', amount=self.amount)

        prune.ln_structured(module=self.model.fc1, name='weight', amount=self.amount, n=self.norme, dim=self.dim)

    def evaluate_sparse_all(self):
        for layer_idx in range(self.nb_layers):
            conv = eval(f"self.model.conv{layer_idx+1}")
            bn = eval(f"self.model.bn{layer_idx+1}")
            self.list_sparse.append(percent_sparse(conv))
            self.list_sparse.append(percent_sparse(bn))

        self.list_sparse.append(percent_sparse(self.model.fc1))

        global_sparse, total_zero_weights, total_weights, df_stats = give_summary_sparse(liste=self.list_sparse, 
                                                                                         list_names=self.list_layers)

        # Reset list for other pruning iterations
        self.list_sparse = []

        return global_sparse, total_zero_weights, total_weights, df_stats



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_filename', type=str, required=True)

    parser.add_argument('--pruning_percent', type=float, default=0.2)
    parser.add_argument('--norme', type=int, default=1)
    parser.add_argument('--iter_pruning', type=int, default=20)
    parser.add_argument('--baseline_acc', type=float, default=0.8)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--frac_data', type=float, default=1.)
    parser.add_argument('--da_mode', type=str, default='')

    args = parser.parse_args()

    TRAINED_MODEL = args.model_filename # (no need to put "models/..")
    MODEL_TYPE = get_model_type(TRAINED_MODEL)  # "ModelA", "ModelB", "ModelC", "ModelD"

    PRUNING_PERCENT = args.pruning_percent      # Prune % of each layer
    NORME = args.norme                          # Ln Norme
    DIMENSION = 0                               # Filters on axis=0
    ITER_PRUNING = args.iter_pruning            # Number of pruning iterations
    BASELINE_ACC = args.baseline_acc            # Accuracy baseline, stop pruning iterations if testAcc is lower than it

    LR = args.lr                                # Learning Rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs                        # Early stopping activated, EPOCHS can be high

    # Misc parameters
    FRAC_DATA = args.frac_data                  # Take {FRAC_DATA}% of the dataset
    DATA_AUGMENT = args.da_mode                 # 'cutmix' or 'random_crop' or 'mixup' or ''
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


    model = load_model(device=DEVICE, saved_model_path=MODEL_PATH+TRAINED_MODEL)
    print(model)
    print(dataset_sizes)
    print(f"Model on device cuda: {next(model.parameters()).is_cuda}")


    ### Define Pruning class ###
    if MODEL_TYPE in ["ModelC", "ModelD"]:
        pruner_class = PrunerResnet(model=model, norme=NORME, amount=PRUNING_PERCENT, dimension=DIMENSION)

    elif MODEL_TYPE in ["ModelA", "ModelB"]:
        nb_layers = 4 if MODEL_TYPE == "ModelA" else 5 # "ModelA": VGG 4 layers | "ModelB": VGG 5 layers
        pruner_class = PrunerVGG(model=model, nb_layers=nb_layers, norme=NORME, amount=PRUNING_PERCENT, dimension=DIMENSION)

    else:
        raise ValueError('Pruning class not defined for this model.')

    print("\n== INFO ==\n")

    print(f"Pruning percent: {PRUNING_PERCENT}, L{NORME} Norm, Pruning iterations: {ITER_PRUNING}, Baseline acc: {BASELINE_ACC*100}%")
    print(f"Epochs: {EPOCHS}, lr: {LR}, Datasets: {FRAC_DATA*100}%")

    print(f"\nBefore pruning:")
    global_sparse, total_zero_weights, total_weights, _ = pruner_class.evaluate_sparse_all()
    test_acc = 0
    best_idx = 0


    ###  Define loss function and optimizer ### 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Define dictionary for training info
    history_training = {'train_loss': [],
                        'val_loss': [],
                        'train_acc': [],
                        'val_acc': []}

    print("Testing before pruning...")
    ### Testing ###
    history_training = test_model(model=model, hist=history_training, criterion=criterion, 
                                  dataloaders=dataloaders, dataset_sizes=dataset_sizes)

    # Pruning
    for idx_prune in range(ITER_PRUNING):
        print(f"\n\nIteration {idx_prune+1} - Pruning {PRUNING_PERCENT*100}% of the least important neurons/filters...")
        pruner_class.prune_all()

        print("\nFine-tuning...\n")
        model, history_training = train_model(model=model, hist=history_training, criterion=criterion, 
                                              optimizer=optimizer, dataloaders=dataloaders, dataset_sizes=dataset_sizes, 
                                              data_augment=DATA_AUGMENT, scheduler=None, num_epochs=EPOCHS)

        ### Testing ###
        history_training = test_model(model=model, hist=history_training, criterion=criterion, 
                                      dataloaders=dataloaders, dataset_sizes=dataset_sizes)

        if history_training['test_acc'] < BASELINE_ACC:
            print(f"STOP PRUNING - test acc below baseline acc ({BASELINE_ACC*100}%)")
            break;

        print("\nAfter pruning and fine-tuning")
        global_sparse, total_zero_weights, total_weights, _ = pruner_class.evaluate_sparse_all()
        test_acc = history_training['test_acc']
        best_idx = idx_prune+1

        # Saving model, and continue pruning
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        saved_model_path = f"{MODEL_PATH}{MODEL_TYPE}_{current_time}_pruned{best_idx}={global_sparse}_testAcc={test_acc}.pth"
        torch.save(model, saved_model_path)
        print(f"Model saved at {saved_model_path}")


    print("\n\n=============================================")
    print("== RESULTS ==\n")

    if test_acc > 0:
        print(f"Pruning successful after {best_idx} iterations.\n")
        print("Best test acc: {:.2f}%\n".format(test_acc))
        print("Global sparsity: {:.2f}%\n".format(global_sparse))
        print("Total number of non-zero parameters: {:d}\n".format(total_weights-total_zero_weights))
        print("Total number of parameters: {:d}".format(total_weights))

    else:
        print("Pruning unsuccessful after one iteration.")

    print("\n=============================================\n\n")
