import json
import os
import sys

import numpy as np
import torch
from numpy.core.defchararray import upper

from configuration.configuration import Configuration
from model.tuning.hyperparam_search import GridSearch
from utils.parser_utils import parse_args

if __name__ == '__main__':
    args, _ = parse_args()

    conf = Configuration(args)
    conf.show_parameters(["param", "bert"])

    if args.dataset_dir is None or len(args.dataset_dir) != 1:
        raise Exception("Train only a dataset per time!")

    if args.saved_model_name is None:
        raise Exception("Define a model name!")

    device = torch.device(conf.gpu)
    print('Python version ', sys.version)
    print('PyTorch version ', torch.__version__)
    print('Device:', device)

    # executing grid search ----> conf.hyperparam_file
    load_checkpoint = False
    if upper(conf.hyperparam_tuning) == 'Y':
        config_file_path = os.path.join("model/tuning", "hyperparameters.json")
    else:
        config_file_path = os.path.join("model/parameters", "model_parameter.json")
        load_checkpoint = True
    with open(config_file_path, 'r') as fp:
        params_of_search = json.load(fp)

    total_number_of_configurations = np.prod([len(val) for val in params_of_search.values()]).item()
    print(f"Grid Search with {total_number_of_configurations} total number of configurations")

    grid_search = GridSearch(params_of_search)
    grid_search.search_and_train(conf=conf, load_checkpoint=load_checkpoint)
