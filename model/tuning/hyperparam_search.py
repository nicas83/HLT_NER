import json
import os
from itertools import product
from typing import Generator, Any

import torch
from transformers import BertModel

from configuration.configuration import Configuration
from model.BERT_CRFClassifier import BERT_CRF_NER
from model.training import train
from utils.custom_dataset import load_datasets


def cross_product(inp: dict):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


class ParameterSequence:
    """
    Base class representing a sequence of parameters for a model-selection
    tuning: accepts a well-formatted input describing configurations for
    the models to be tested and returns each time the corresponding model,
    optimizer, loss etc.
    """

    def get_configs(self, data) -> Generator[dict, Any, None]:
        pass


class ParameterGrid(ParameterSequence):
    """
    A grid of parameters s.t. each possible combination of each value for
    each parameter is a candidate configuration.
    """

    def get_configs(self, data) -> Generator[dict, Any, None]:
        hyperpar_comb = cross_product(data)
        for comb in hyperpar_comb:
            yield comb


class ParameterList(ParameterSequence):
    """
    A list of "already-compiled" configurations to be tested all.
    """

    def get_configs(self, data) -> Generator[dict, Any, None]:
        for config in data:
            yield config


class BaseSearch:
    """
    Base class representing a tuning over a hyperparameter(s) space.
    """

    def __init__(self, parameters):
        """
        :param parameters: An object (dict, list, ..., depending on the specific class)
        that is used for generating all the needed configurations.
        :param scoring: A metric used to evaluate each configuration performance and sort
        all them at the end.
        :param cross_validator: Validator to be used for splitting and iterating over
        training and validation data, e.g. Holdout or KFold. Defaults to None.
        """
        self.parameters = parameters
        self.results = []  # results of each grid tuning case

    def setup_parameters(self) -> ParameterSequence:
        pass

    def search_and_train(self, conf: Configuration, load_checkpoint=False):

        datasets, label_handler, data_processor = load_datasets(conf)
        bert_model = BertModel.from_pretrained(conf.bert, use_cache=False)
        model = BERT_CRF_NER(bert_model, label_handler.get_start_label_id(), label_handler.get_stop_label_id(),
                             len(label_handler.get_label_list()), conf.gpu)
        if load_checkpoint:
            checkpoint = torch.load(conf.path_saved_model + conf.saved_model_name, map_location=conf.gpu)
            pretrained_dict = checkpoint['model_state']
            net_state_dict = model.state_dict()
            pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
            net_state_dict.update(pretrained_dict_selected)
            model.load_state_dict(net_state_dict)
            model.to(torch.device(conf.gpu))

        parameters_sequence = self.setup_parameters()
        hyperpar_comb = parameters_sequence.get_configs(self.parameters)
        print('Retrieve parameters combination')
        for i, comb in enumerate(hyperpar_comb):
            print("***** Number of hyper-parameter combination: ", i + 1)
            self.results.append(train(model, datasets.get_train_data(), datasets.get_dev_data(), conf, comb,
                                      data_processor=data_processor))

        self.results = sorted(self.results, key=lambda x: x['f1_score'])
        self.save_all()

    def save_all(self, directory_path: str = '.', file_name: str = 'results.json'):
        number = len(self.results)
        self.save_best(number, directory_path, file_name)

    def save_best(
            self, number: int, directory_path: str = '.',
            file_name: str = 'best_results.json'):
        results = self.results[:number]
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'w') as fp:
            json.dump(results, fp, indent=2)


class GridSearch(BaseSearch):
    """
    Grid Search: __init__ method accepts a dict of the form {str:list}
    as first parameter and cycles over all the possible combinations.
    """

    def setup_parameters(self) -> ParameterSequence:
        return ParameterGrid()
