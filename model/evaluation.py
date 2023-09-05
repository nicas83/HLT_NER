import time

import numpy as np
import torch
from pandas import DataFrame
from torch import zeros
from tqdm import tqdm

from configuration.configuration import Configuration
from utils.metrics import f1_score, DictErrors, scores


def evaluate(model, predict_dataloader, epoch_th, dataset_name, conf: Configuration, data_processor=None,
             return_dict: bool = False):

    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    dict_errors = DictErrors() if return_dict else None
    start = time.time()
    confusion = zeros(size=(len(data_processor.get_label_map()), len(data_processor.get_label_map())))
    with torch.no_grad():
        for batch in tqdm(predict_dataloader, desc="Validation", mininterval=conf.refresh_rate):
            batch = tuple(t.to(conf.gpu) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            _, predicted_label_seq_ids = model(input_ids, input_mask)
            valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)

            for lbl, pre in zip(valid_label_ids, valid_predicted):
                confusion[lbl, pre] += 1

            if dict_errors is not None:
                dict_errors.add(input_ids, predicted_label_seq_ids, label_ids)

            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct / total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend:%.3f minutes for evaluation' \
          % (epoch_th, 100. * test_acc, 100. * precision, 100. * recall, 100. * f1, dataset_name, (end - start) / 60.0))
    print('--------------------------------------------------------------')

    output_results = scores(confusion, all_metrics=True)
    output_results.index = data_processor.map_id2lab([*range(0, len(data_processor.get_label_map()))])
    print(output_results)
    output_results.to_json('output_result_test_evaluation.json', orient='records')

    output = DataFrame({
        "Accuracy": test_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1}, index=['accuracy', 'precision', 'recall', 'f1'])
    print(output)
    return output, dict_errors.result() if return_dict else None

