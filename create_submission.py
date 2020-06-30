import time
import tqdm
import argparse
import numpy as np
import pandas as pd

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
from torch.nn.functional import softmax

from create_dataset import create_eval_dataset
from create_dataset import DATA_PATH, MODEL_PATH, CSV_PATH, DICT_ENCODE_CLASS, DEVICE
from utils import load_model



def eval_model(model, criterion, evalloader, evalloader_length, half=False):
    """
    Evaluation function. 
    Returns the inference output.
    """

    model.eval()   # Set model to evaluate mode

    sincetime = time.time()
    outputs_list = []
    list_probs = []

    pbar = tqdm.tqdm([i for i in range(evalloader_length)])

    for batch_idx, inputs in enumerate(evalloader):
        inputs = inputs.to(DEVICE)

        if half:
            inputs=inputs.half()

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            probs = softmax(outputs,1)
            outputs_list.append(probs.cpu().numpy()[0])

        list_probs.append(probs.cpu())

        pbar.update()
        pbar.set_description("Processing audio %s" % str(batch_idx+1))      

    pbar.close()
    print("Inference on EvaluationSet completed in {:.1f}s\n".format(time.time() - sincetime))

    return outputs_list


def fill_submission(df, outputs_list, submission_path):
    """
    Fill info to dataframe and save to csv.
    """

    scene_list = [DICT_ENCODE_CLASS[np.argmax(output)] for output in outputs_list]
    indoor_list = [output[1] for output in outputs_list]
    outdoor_list = [output[0] for output in outputs_list]
    transportation_list = [output[2] for output in outputs_list]

    df['scene_label'] = scene_list
    df['indoor'] = indoor_list
    df['outdoor'] = outdoor_list
    df['transportation'] = transportation_list
    print("Final Evaluation df shape: ", df.shape)

    df = df.reset_index(drop=True)

    df.to_csv(submission_path, sep='\t')
    print(f"Submission saved at {submission_path}.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_filename', type=str, required=True)
    parser.add_argument('--eval_csv_path', type=str, required=True)
    parser.add_argument('--eval_data_path', type=str, required=True)
    parser.add_argument('--submission_filename', type=str, required=False, default="submission.csv")
    parser.add_argument('--half', type=int, required=False, default=0)

    args = parser.parse_args()

    TRAINED_MODEL = args.model_filename
    EVAL_CSV_PATH = args.eval_csv_path          # "evaluation_setup/fold1_evaluation.csv"
    EVAL_DATA_PATH = args.eval_data_path        # "audio_S18k/"
    SUBMISSION_NAME = args.submission_filename  # "submission.csv"
    HALF = args.half                            # int 0: no, 1: yes (if the model has been halfed)

    # Load model
    model = load_model(device=DEVICE, saved_model_path=MODEL_PATH+TRAINED_MODEL)


    # Create dataloader
    df_eval = pd.read_csv(EVAL_CSV_PATH) 
    print("Evaluation df shape: ", df_eval.shape)
    evalloader_length = create_eval_dataset(df_eval, eval_data_path=EVAL_DATA_PATH, frac_data=1, random_state=17)

    evalloader = torch.load(f'{DATA_PATH}evaluation_data.pt')


    # Inference
    criterion = nn.CrossEntropyLoss()

    outputs_list = eval_model(model=model, criterion=criterion, evalloader=evalloader, 
                              evalloader_length=evalloader_length, half=HALF)

    # Create submission csv
    fill_submission(df=df_eval, outputs_list=outputs_list, submission_path=CSV_PATH+SUBMISSION_NAME)






