import sys

import torch
from pandas import DataFrame
from torch.utils import data
from transformers import BertModel

from configuration.configuration import Configuration
from model.BERT_CRFClassifier import BERT_CRF_NER
from model.evaluation import evaluate
from utils.custom_dataset import load_datasets, NerDataset
from utils.parser_utils import parse_args


def outputs(results, error_dict):
    if isinstance(results, DataFrame):
        results.to_json('output_test_evaluation.json', orient='records')
    else:
        print(results.getvalue())

    if error_dict is not None:
        print(error_dict)


if __name__ == '__main__':

    args, _ = parse_args()

    conf = Configuration(args)

    if args.dataset_dir is None or len(args.dataset_dir) != 1:
        raise Exception("Train only a dataset per time!")

    if args.saved_model_name is None:
        raise Exception("Define a model name!")

    device = torch.device(conf.gpu)
    print('Python version ', sys.version)
    print('PyTorch version ', torch.__version__)
    print('Device:', device)

    paths = args.dataset_dir
    models = args.saved_model_name

    # load the test set
    ''' Prepare data set '''
    datasets, label_handler, data_processor = load_datasets(conf, evaluate=True)

    # load the saved model
    bert_model = BertModel.from_pretrained(conf.bert)
    model = BERT_CRF_NER(bert_model, label_handler.get_start_label_id(), label_handler.get_stop_label_id(),
                         len(label_handler.get_unique_labels()), device)
    checkpoint = torch.load(conf.path_saved_model + conf.saved_model_name, map_location='cpu')
    epoch = checkpoint['epoch']
    pretrained_dict = checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)

    model.to(device)
    test_dataloader = data.DataLoader(dataset=datasets.get_test_data(), batch_size=checkpoint['batch_size'],
                                      shuffle=False, collate_fn=NerDataset.pad)

    output_results, error_dict = evaluate(model, test_dataloader, epoch - 1, 'Test_set', conf,
                                          data_processor=data_processor)

    outputs(output_results, error_dict)
