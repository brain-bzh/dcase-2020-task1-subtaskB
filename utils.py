import datetime
import numpy as np

import torch

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import models



# Dictionary of all the models
DICT_MODELS = {
              "ModelA": models.ModelA,
              "ModelB": models.ModelB,
              "ModelC": models.ModelC,
              "ModelD": models.ModelD,
              }


def rand_cut(size, lam):
    """
    Return the indexes of the random cut for mixup or cutmix.
    """
    audio_length = size[2]
    
    cut_ratio = 1. - lam
    cut_length = np.int(audio_length * cut_ratio)
    # uniform
    cut_rand = np.random.randint(audio_length)
    cut1 = np.clip(cut_rand - cut_length // 2, 0, audio_length)
    cut2 = np.clip(cut_rand + cut_length // 2, 0, audio_length)
    
    return (cut1, cut2)


def load_model(device, model_type="", saved_model_path=None):
    """
    Load and return model.
    - saved_model_path: Path to a trained model .pth
    """

    if model_type:
        model = DICT_MODELS[model_type]()
        print(f"{model_type} loaded.")

    else:
        model = torch.load(saved_model_path)
        print(f"{saved_model_path} loaded.")

    model.to(device)

    return model


def save_model(model, hist, trained_models_path, model_type, do_save):
    """
    Save the trained model.
    """
    if do_save:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        saved_model_path = f"{trained_models_path}{model_type}_{current_time}_trained_testAcc={hist['test_acc']}.pth"
        torch.save(model, saved_model_path)
        print(f"Model saved at {saved_model_path}")


def plot_training(hist, graphs_path, model_type, do_save, do_plot=False):
    """
    Plot the training and validation loss/accuracy.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title(f'{model_type} - loss')
    ax[0].plot(hist["epochs"], hist["train_loss"], label="Train loss")
    ax[0].plot(hist["epochs"], hist["val_loss"], label="Validation loss")
    ax[1].set_title(f'{model_type} - accuracy')
    ax[1].plot(hist["epochs"], hist["train_acc"], label="Train accuracy")
    ax[1].plot(hist["epochs"], hist["val_acc"], label="Validation accuracy")
    ax[0].legend()
    ax[1].legend()
    if do_save:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_graph_path = f"{graphs_path}{model_type}_losses&acc_{current_time}_testAcc={hist['test_acc']}.png"
        plt.savefig(save_graph_path)
        print(f"Training graph saved at {save_graph_path}")
    if do_plot: plt.show()


def classif_report(hist, list_names=[]):
    """
    Give the classification report from sklearn.
    """
    y_pred = [y for y in hist['y_pred']]
    y_true = [y for y in hist['y_true']]

    nb_classes = len(set(y_true))

    accuracy = round(accuracy_score(y_true, y_pred)*100, 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    print(f'Accuracy: {accuracy}%')
    print(f'MSE: {mse}')
    target_names = list_names if list_names else [f'class {i}' for i in range(nb_classes)]
    print(classification_report(y_true, y_pred, target_names=target_names))


def plot_cm(hist, graphs_path, model_type, do_save, do_plot=False):
    """
    Plot the confusion matrix after testing.
    """
    y_pred = [y for y in hist['y_pred']]
    y_true = [y for y in hist['y_true']]

    nb_classes = len(set(y_true))
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cm_path = f"{graphs_path}{model_type}_CM_{current_time}_testAcc={hist['test_acc']}.png"

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index = [i for i in range(nb_classes)], 
                         columns = [i for i in range(nb_classes)])
    plt.figure(figsize = (10,7))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.heatmap(df_cm, cmap=cmap, annot=True)
    plt.title(f"Confusion Matrix for {model_type}")

    if do_save:
        plt.savefig(cm_path)
        print(f"Confusion Matrix saved at {cm_path}")
    if do_plot: plt.show()


# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='models/checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def half_model(model):
    """
    Return the quantified model.
    """
    print("\nQuantify model\n")
    model = model.half()

    return model


def calculate_loss(hist, list_names):
    """
    Print log_loss from sklearn.
    """
    y_true = hist['y_true']
    y_probs = hist['probs']

    list_ce_loss = []
    indexes = {i: [idx for idx, value in enumerate(y_true) if value == i] for i in range(len(list_names))}

    for class_idx, class_name in enumerate(list_names):
        print("\nClass:", class_name)
        new_y_true = [[1 if i == value else 0 for i in range(len(list_names)) ] 
                      for idx, value in enumerate(y_true) if idx in indexes[class_idx]]
        new_y_probs = y_probs[indexes[class_idx]]

        ce_loss = log_loss(new_y_true, new_y_probs, eps=1e-7) # eps=1e-7 in case the model is quantified
        list_ce_loss.append((ce_loss, len(new_y_true)))

        print('CE:', ce_loss)

    print('\nTotal CE:', log_loss(y_true, y_probs, eps=1e-7))

    print('\nTotal CE (sanity check):', np.sum([ce_loss * ponderation for ce_loss, ponderation in list_ce_loss])\
          /np.sum([ponderation for ce_loss, ponderation in list_ce_loss]))


def get_model_type(model_path):
    """
    Return the model type depending on the name of the model path.
    """
    for model in DICT_MODELS.keys():
        if model in model_path:
            return model
    raise ValueError('Model type not identified from the trained_model_path params.'
                     +'\nPlease check that either ModelA, ModelB, ModelC or ModelD is present in the filename.')
